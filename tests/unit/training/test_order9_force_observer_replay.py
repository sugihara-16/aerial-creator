from __future__ import annotations

import torch

from amsrr.training.order9_anchor_normal_force_estimator import (
    Order9AnchorNormalForceEstimatorConfig,
)
from amsrr.training.order9_force_observer_replay import (
    ORDER9_FORCE_OBSERVER_TRACE_VERSION,
    compare_order9_force_observer_baseline,
    load_order9_force_observer_trace,
    write_order9_force_observer_trace,
)


def _trace() -> dict[str, object]:
    step_count, batch, anchors, joints = 30, 1, 2, 2
    jacobian = torch.zeros(step_count, batch, anchors, 3, joints)
    jacobian[:, :, 0, 0, 0] = 0.10
    jacobian[:, :, 1, 0, 1] = 0.10
    reaction = torch.zeros(step_count, batch, anchors, 3)
    reaction[..., 0] = 1.0
    baseline = torch.tensor([0.40, 0.40]).reshape(1, 1, joints)
    applied = baseline.expand(step_count, batch, joints).clone()
    expected_force = torch.tensor([5.0, 6.0]).reshape(1, 1, anchors)
    normal_jacobian = -torch.einsum(
        "tbax,tbaxj->tbaj", reaction, jacobian
    )
    contact_torque = torch.einsum(
        "tbaj,tba->tbj",
        normal_jacobian[10:],
        expected_force.expand(step_count - 10, batch, anchors),
    )
    applied[10:] += contact_torque
    actual = torch.zeros(step_count, batch, anchors)
    actual[10:] = expected_force
    selected = torch.ones(step_count, batch, anchors, dtype=torch.bool)
    config = Order9AnchorNormalForceEstimatorConfig(
        force_filter_alpha=1.0,
        ridge_damping_m2=1.0e-8,
        projected_iterations=12,
        fit_residual_scale_nm=1.0,
    )
    return {
        "trace_version": ORDER9_FORCE_OBSERVER_TRACE_VERSION,
        "metadata": {
            "force_estimator_config": {
                name: getattr(config, name)
                for name in config.__dataclass_fields__
            },
            "control_dt_s": 0.02,
            "confidence_threshold": 0.25,
            "contact_distance_tolerance_m": 0.004,
            "contact_inward_limit_m": 0.006,
            "relative_speed_tolerance_mps": 0.05,
            "contact_dwell_s": 0.20,
        },
        "tensors": {
            "applied_joint_torque_nm": applied,
            "gravity_joint_torque_nm": torch.zeros_like(applied),
            "grasp_point_linear_jacobian_world": jacobian,
            "reaction_normal_world": reaction,
            "anchor_joint_owner_mask": torch.eye(joints, dtype=torch.bool),
            "selected_anchor_mask": selected,
            "baseline_update_mask": torch.arange(step_count)[:, None] < 10,
            "estimation_active_mask": torch.arange(step_count)[:, None] >= 10,
            "privileged_actual_normal_force_n": actual,
            "required_normal_force_n": torch.full_like(actual, 4.0),
            "surface_distance_m": torch.zeros_like(actual),
            "relative_speed_mps": torch.zeros_like(actual),
            "qp_feasible": torch.ones(step_count, batch, dtype=torch.bool),
            "phase_index": torch.cat(
                (
                    torch.zeros(10, batch, dtype=torch.long),
                    torch.ones(20, batch, dtype=torch.long),
                )
            ),
            "episode_serial": torch.zeros(step_count, batch, dtype=torch.long),
            "online_normal_force_n": torch.zeros_like(actual),
            "online_confidence": torch.zeros_like(actual),
        },
    }


def test_offline_baseline_replay_recovers_contact_increment() -> None:
    comparison = compare_order9_force_observer_baseline(_trace())
    with_baseline = comparison["with_baseline"]
    without_baseline = comparison["without_baseline"]
    assert with_baseline["mean_absolute_error_n"] < 1.0e-3
    assert without_baseline["mean_absolute_error_n"] > 3.0
    assert with_baseline["contact_dwell_admitted_environment_count"] == 1
    assert without_baseline["contact_dwell_admitted_environment_count"] == 0


def test_geometry_gated_branch_baseline_tracks_contact_phase_free_motion() -> None:
    trace = _trace()
    tensors = trace["tensors"]
    metadata = trace["metadata"]
    metadata["force_estimator_config"]["baseline_update_alpha"] = 1.0
    # Contact acquisition starts before physical contact.  A closing command
    # changes the free-motion torque while both anchors remain 10 mm clear.
    tensors["surface_distance_m"][10:15] = 0.010
    tensors["privileged_actual_normal_force_n"][10:15] = 0.0
    tensors["applied_joint_torque_nm"][10:15] = torch.tensor([0.10, 0.10])
    tensors["applied_joint_torque_nm"][15:] = torch.tensor([-0.40, -0.50])
    tensors["privileged_actual_normal_force_n"][15:] = torch.tensor([5.0, 6.0])
    comparison = compare_order9_force_observer_baseline(trace)
    phase_only = comparison["phase_only_baseline"]
    geometry_gated = comparison["geometry_gated_branch_baseline"]
    assert phase_only["mean_absolute_error_n"] > 2.5
    assert geometry_gated["mean_absolute_error_n"] < 1.0e-3
    assert geometry_gated["contact_dwell_admitted_environment_count"] == 1


def test_force_observer_trace_roundtrip(tmp_path) -> None:
    trace = _trace()
    target = tmp_path / "trace.pt"
    digest = write_order9_force_observer_trace(
        target,
        metadata=trace["metadata"],
        tensors=trace["tensors"],
    )
    assert len(digest) == 64
    loaded = load_order9_force_observer_trace(target)
    torch.testing.assert_close(
        loaded["tensors"]["applied_joint_torque_nm"],
        trace["tensors"]["applied_joint_torque_nm"],
    )
