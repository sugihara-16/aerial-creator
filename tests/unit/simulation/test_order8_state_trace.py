from __future__ import annotations

import copy

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.order8_state_trace import (
    build_order8_state_trace,
    load_order8_state_trace,
    validate_order8_state_trace,
    write_order8_state_trace,
)


def _frame(time_s: float, position: float) -> dict[str, object]:
    return {
        "simulation_time_s": time_s,
        "phase": "contact_acquisition",
        "modules": {
            "0": {
                "root_pose_world": [position, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "root_twist_world": [0.0] * 6,
                "joint_positions_rad": [position, -position],
                "joint_velocities_radps": [0.1, -0.1],
            }
        },
        "object_pose_world": [0.5, 0.0, 0.075, 0.0, 0.0, 0.0, 1.0],
        "object_twist_world": [0.0] * 6,
    }


def _trace() -> dict[str, object]:
    return build_order8_state_trace(
        simulation_dt_s=0.01,
        frame_stride=2,
        graph_id="graph",
        graph_hash="graph-hash",
        config_hash="config-hash",
        source_urdf_sha256="urdf-hash",
        generated_usd_sha256="usd-hash",
        module_ids=[0],
        joint_names_by_module={0: ["joint_a", "joint_b"]},
        source_probe_argv=[
            "--order8-natural-contact",
            "--order8-diagnostic-only",
        ],
        frames=[_frame(0.0, 0.0), _frame(0.02, 0.1)],
    )


def test_state_trace_round_trip_is_hash_bound_and_acceptance_ineligible(tmp_path) -> None:
    trace = _trace()
    output_path = write_order8_state_trace(tmp_path / "trace.json", trace)

    restored = load_order8_state_trace(output_path)

    assert restored == trace
    assert restored["diagnostic_only"] is True
    assert restored["acceptance_eligible"] is False
    assert restored["replay_advances_physics"] is False


def test_state_trace_rejects_tampered_frame() -> None:
    trace = copy.deepcopy(_trace())
    trace["frames"][1]["object_pose_world"][0] = 2.0  # type: ignore[index]

    with pytest.raises(SchemaValidationError, match="payload hash mismatch"):
        validate_order8_state_trace(trace)


def test_state_trace_rejects_non_monotonic_time_before_hash_check() -> None:
    trace = copy.deepcopy(_trace())
    trace["frames"][1]["simulation_time_s"] = 0.0  # type: ignore[index]

    with pytest.raises(SchemaValidationError, match="strictly increasing"):
        validate_order8_state_trace(trace)
