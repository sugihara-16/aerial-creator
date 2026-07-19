from __future__ import annotations

"""Diagnostic-only state traces for wall-clock Order-8 GUI replay.

The trace deliberately records kinematic simulator state, not contact forces or
acceptance evidence.  A replay writes the recorded poses and joint states into
Isaac without advancing physics.  It is therefore useful for visual inspection
only and can never satisfy the natural-contact smoke contract.
"""

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from amsrr.schemas.common import SchemaValidationError, canonical_json
from amsrr.utils.hashing import stable_hash


ORDER8_STATE_TRACE_VERSION = "order8_diagnostic_state_trace_v1"


def build_order8_state_trace(
    *,
    simulation_dt_s: float,
    frame_stride: int,
    graph_id: str,
    graph_hash: str,
    config_hash: str,
    source_urdf_sha256: str,
    generated_usd_sha256: str,
    module_ids: Sequence[int],
    joint_names_by_module: Mapping[int, Sequence[str]],
    source_probe_argv: Sequence[str],
    frames: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build and validate one immutable diagnostic trace payload."""

    payload: dict[str, object] = {
        "schema_version": ORDER8_STATE_TRACE_VERSION,
        "diagnostic_only": True,
        "acceptance_eligible": False,
        "replay_advances_physics": False,
        "simulation_dt_s": float(simulation_dt_s),
        "frame_stride": int(frame_stride),
        "graph_id": str(graph_id),
        "graph_hash": str(graph_hash),
        "config_hash": str(config_hash),
        "source_urdf_sha256": str(source_urdf_sha256),
        "generated_usd_sha256": str(generated_usd_sha256),
        "module_ids": [int(module_id) for module_id in module_ids],
        "joint_names_by_module": {
            str(int(module_id)): [str(name) for name in names]
            for module_id, names in joint_names_by_module.items()
        },
        "source_probe_argv": [str(value) for value in source_probe_argv],
        "frames": [dict(frame) for frame in frames],
    }
    payload["trace_payload_hash"] = _trace_payload_hash(payload)
    validate_order8_state_trace(payload)
    return payload


def validate_order8_state_trace(payload: Mapping[str, object]) -> None:
    """Fail closed on malformed, tampered, or acceptance-mislabelled traces."""

    if payload.get("schema_version") != ORDER8_STATE_TRACE_VERSION:
        raise SchemaValidationError("unsupported Order8 state-trace version")
    if payload.get("diagnostic_only") is not True:
        raise SchemaValidationError("Order8 state trace must be diagnostic-only")
    if payload.get("acceptance_eligible") is not False:
        raise SchemaValidationError("Order8 state trace cannot be acceptance eligible")
    if payload.get("replay_advances_physics") is not False:
        raise SchemaValidationError("Order8 state replay must not advance physics")

    simulation_dt_s = _finite_number(payload.get("simulation_dt_s"), "simulation_dt_s")
    if simulation_dt_s <= 0.0:
        raise SchemaValidationError("simulation_dt_s must be positive")
    frame_stride = payload.get("frame_stride")
    if not isinstance(frame_stride, int) or isinstance(frame_stride, bool) or frame_stride <= 0:
        raise SchemaValidationError("frame_stride must be a positive integer")

    for name in (
        "graph_id",
        "graph_hash",
        "config_hash",
        "source_urdf_sha256",
        "generated_usd_sha256",
    ):
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            raise SchemaValidationError(f"{name} must be a non-empty string")

    module_ids_raw = payload.get("module_ids")
    if not isinstance(module_ids_raw, list) or not module_ids_raw:
        raise SchemaValidationError("module_ids must be a non-empty list")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in module_ids_raw
    ):
        raise SchemaValidationError("module_ids must contain non-negative integers")
    module_ids = [int(value) for value in module_ids_raw]
    if module_ids != sorted(set(module_ids)):
        raise SchemaValidationError("module_ids must be sorted and unique")
    module_keys = {str(value) for value in module_ids}

    joint_names_raw = payload.get("joint_names_by_module")
    if not isinstance(joint_names_raw, dict) or set(joint_names_raw) != module_keys:
        raise SchemaValidationError(
            "joint_names_by_module must exactly cover the recorded modules"
        )
    joint_names_by_module: dict[str, list[str]] = {}
    for module_key, names_raw in joint_names_raw.items():
        if not isinstance(names_raw, list):
            raise SchemaValidationError("joint-name rows must be lists")
        if any(not isinstance(name, str) or not name for name in names_raw):
            raise SchemaValidationError("joint names must be non-empty strings")
        names = [str(name) for name in names_raw]
        if len(names) != len(set(names)):
            raise SchemaValidationError("joint names must be unique per module")
        joint_names_by_module[str(module_key)] = names

    source_probe_argv = payload.get("source_probe_argv")
    if not isinstance(source_probe_argv, list) or any(
        not isinstance(value, str) for value in source_probe_argv
    ):
        raise SchemaValidationError("source_probe_argv must be a string list")

    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise SchemaValidationError("Order8 state trace requires at least two frames")
    previous_time_s = -math.inf
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise SchemaValidationError(f"frame {index} must be an object")
        time_s = _finite_number(frame.get("simulation_time_s"), "simulation_time_s")
        if time_s < 0.0 or time_s <= previous_time_s:
            raise SchemaValidationError("frame times must be finite and strictly increasing")
        previous_time_s = time_s
        phase = frame.get("phase")
        if not isinstance(phase, str) or not phase:
            raise SchemaValidationError(f"frame {index} phase must be non-empty")
        modules = frame.get("modules")
        if not isinstance(modules, dict) or set(modules) != module_keys:
            raise SchemaValidationError(
                f"frame {index} modules must exactly cover module_ids"
            )
        for module_key, state in modules.items():
            if not isinstance(state, dict):
                raise SchemaValidationError(
                    f"frame {index} module {module_key} state must be an object"
                )
            _finite_vector(state.get("root_pose_world"), 7, "root_pose_world")
            _unit_quaternion(state["root_pose_world"], "root_pose_world")
            _finite_vector(state.get("root_twist_world"), 6, "root_twist_world")
            joint_count = len(joint_names_by_module[str(module_key)])
            _finite_vector(
                state.get("joint_positions_rad"),
                joint_count,
                "joint_positions_rad",
            )
            _finite_vector(
                state.get("joint_velocities_radps"),
                joint_count,
                "joint_velocities_radps",
            )
        _finite_vector(frame.get("object_pose_world"), 7, "object_pose_world")
        _unit_quaternion(frame["object_pose_world"], "object_pose_world")
        _finite_vector(frame.get("object_twist_world"), 6, "object_twist_world")

    recorded_hash = payload.get("trace_payload_hash")
    if not isinstance(recorded_hash, str) or recorded_hash != _trace_payload_hash(payload):
        raise SchemaValidationError("Order8 state-trace payload hash mismatch")


def write_order8_state_trace(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Validate and atomically write a trace JSON artifact."""

    validate_order8_state_trace(payload)
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def load_order8_state_trace(path: str | Path) -> dict[str, object]:
    input_path = Path(path).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SchemaValidationError("Order8 state trace root must be an object")
    validate_order8_state_trace(payload)
    return payload


def _trace_payload_hash(payload: Mapping[str, object]) -> str:
    return stable_hash(
        {key: value for key, value in payload.items() if key != "trace_payload_hash"}
    )


def _finite_number(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise SchemaValidationError(f"{label} must be finite")
    return float(value)


def _finite_vector(value: object, length: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise SchemaValidationError(f"{label} must have length {length}")
    for item in value:
        _finite_number(item, label)


def _unit_quaternion(pose: object, label: str) -> None:
    assert isinstance(pose, list)
    norm = math.sqrt(sum(float(value) ** 2 for value in pose[3:7]))
    if abs(norm - 1.0) > 1.0e-3:
        raise SchemaValidationError(f"{label} quaternion must be unit length")
