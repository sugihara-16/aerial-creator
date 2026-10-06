"""Independent physical-motion audit with one archive read per environment batch.

The predicate and report match the frozen 20260917 audit. Only archive loading
and hashing are shared; every environment is checked from its own raw tensors.
"""
from pathlib import Path
import json

import torch
from amsrr.utils.hashing import hash_file


def audit_all(directory):
    directory = Path(directory)
    raw_path = directory / "rollout.pt"
    raw = torch.load(raw_path, map_location="cpu", weights_only=True, mmap=True)
    result = json.loads((directory / "result.json").read_text())
    scene = json.loads((directory / "scene.json").read_text())
    digest = hash_file(raw_path)
    return [_audit(directory, environment, raw, result, scene, digest)
            for environment in range(len(result["episodes"]))]


def _audit(directory, environment, raw, result, scene, digest):
    data = raw["tensors"]
    origins = result.get("environment_origins_world", [[0., 0., 0.]])
    origin = torch.tensor(origins[environment], dtype=torch.float64)
    valid = data["valid"][:, environment].bool()
    pose = data["post_object_pose_world"][valid, environment].double().clone()
    pose[:, :3] -= origin
    initial = data["object_pose_world"][valid, environment][0].double().clone()
    initial[:3] -= origin
    twist = data["post_object_twist_world"][valid, environment].double()
    goal = next(g for g in scene["task"]["goals"] if g["goal_type"] == "object_pose")
    target = torch.tensor(goal["target_pose_world"], dtype=torch.float64)
    error = torch.linalg.vector_norm(pose[:, :3] - target[:3], dim=-1)
    q = torch.nn.functional.normalize(pose[:, 3:], dim=-1)
    desired_q = torch.nn.functional.normalize(target[3:], dim=-1)
    angle = 2 * torch.acos((q * desired_q).sum(-1).abs().clamp(max=1.0))
    reached = (error <= goal["tolerance_pos_m"]) & (
        angle <= goal["tolerance_rot_rad"]
    )
    dt = float(raw["metadata"]["control_dt_s"])
    times = data["time_s"][valid, environment].double() + dt
    first_reach = float(times[torch.nonzero(reached)[0, 0]]) if reached.any() else None
    success = (result["passed"] if len(result["episodes"]) == 1
               else result["episodes"][environment]["task_success"])
    report = {
        "physical_rollout_sha256": digest,
        "valid_steps": int(valid.sum()),
        "initial_position_m": initial[:3].tolist(),
        "final_position_m": pose[-1, :3].tolist(),
        "maximum_lift_m": float((pose[:, 2] - initial[2]).max()),
        "maximum_xy_transport_m": float(
            torch.linalg.vector_norm(pose[:, :2] - initial[:2], dim=-1).max()
        ),
        "final_position_error_m": float(error[-1]),
        "final_orientation_error_rad": float(angle[-1]),
        "final_linear_speed_mps": float(torch.linalg.vector_norm(twist[-1, :3])),
        "final_angular_speed_radps": float(torch.linalg.vector_norm(twist[-1, 3:])),
        "first_pose_goal_reach_s": first_reach,
        "authored_pose_goal_time_limit_s": goal["time_limit_s"],
        "pose_goal_reached_within_authored_time_limit": (
            first_reach is not None and first_reach <= goal["time_limit_s"]
        ),
        "episode_simulation_seconds": float(times[-1]),
        "maximum_pi_l_global_residual": float(data["applied_global_action"][valid, environment].abs().max()),
        "maximum_pi_l_contact_residual": float(
            data["contact_space_residual_action"][valid, environment].abs().max()
        ),
        "prohibited_collision": bool(data["prohibited_collision"][valid, environment].any()),
        "runtime_reported_success": success,
        "environment": environment,
    }
    report["physical_goal_and_motion_passed"] = bool(
        reached[-1]
        and report["maximum_lift_m"] >= 0.048
        and report["maximum_xy_transport_m"] >= 0.15
        and report["final_linear_speed_mps"] <= 0.05
        and report["final_angular_speed_radps"] <= 0.1
        and not report["prohibited_collision"]
        and report["pose_goal_reached_within_authored_time_limit"]
    )
    if success and not report["physical_goal_and_motion_passed"]:
        raise ValueError("runtime success disagrees with independent physical audit")
    destination = directory / ("independent_motion.json" if len(result["episodes"]) == 1
                               else f"independent_motion_env_{environment}.json")
    if destination.exists():
        if json.loads(destination.read_text()) != report:
            raise ValueError("saved physical audit changed")
        return report
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report

