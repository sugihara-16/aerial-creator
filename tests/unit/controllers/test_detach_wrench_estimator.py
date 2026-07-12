from __future__ import annotations

import pytest

from amsrr.controllers.detach_wrench_estimator import (
    CUT_WRENCH_SIGN_CONVENTION,
    DetachUnloadGate,
    DetachUnloadGateConfig,
    DetachWrenchEstimate,
    FollowerSubtreeDetachWrenchEstimator,
    transform_follower_com_wrench_to_dock_frame,
)
from amsrr.morphology.dock_geometry import relative_pose_for_dock_ports
from amsrr.robot_model.physical_model_builder import (
    build_module_capability_token,
    build_physical_model_from_config,
)
from amsrr.schemas.morphology import DockEdge, ModuleNode, MorphologyGraph, PortNode
from amsrr.schemas.policies import ControllerCommand, ControllerStatus
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState


def _case():
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    capability = build_module_capability_token(physical_model)
    src_spec = physical_model.dock_ports[0]
    dst_spec = next(
        item
        for item in physical_model.dock_ports
        if item.port_type in src_spec.compatible_port_types
    )
    src_port = PortNode(
        port_global_id=0,
        module_id=0,
        port_local_id=src_spec.port_id,
        local_pose=src_spec.local_pose,
        port_type=src_spec.port_type,
        occupied=True,
        compatible_port_type_mask=[],
    )
    dst_port = PortNode(
        port_global_id=1,
        module_id=1,
        port_local_id=dst_spec.port_id,
        local_pose=dst_spec.local_pose,
        port_type=dst_spec.port_type,
        occupied=True,
        compatible_port_type_mask=[],
    )
    relative_pose = relative_pose_for_dock_ports(src_port, dst_port)
    graph = MorphologyGraph(
        graph_id="detach-estimator-test",
        modules=[
            ModuleNode(
                module_id=0,
                module_type="holon",
                pose_in_design_frame=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                role_id="base",
                is_base=True,
                capability_token=capability,
            ),
            ModuleNode(
                module_id=1,
                module_type="holon",
                pose_in_design_frame=relative_pose,
                role_id="follower",
                is_base=False,
                capability_token=capability,
            ),
        ],
        ports=[src_port, dst_port],
        dock_edges=[
            DockEdge(
                edge_id=0,
                src_module_id=0,
                src_port_id=0,
                dst_module_id=1,
                dst_port_id=1,
                relative_pose_src_to_dst=relative_pose,
                edge_role="structural",
                estimated_stiffness=[1.0] * 6,
                latch_state="attached",
            )
        ],
        robot_anchors=[],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )
    observation = RuntimeObservation(
        time_s=0.1,
        morphology_graph=graph,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module.pose_in_design_frame,
                twist_world=[0.0] * 6,
                joint_positions={},
                joint_velocities={},
            )
            for module in graph.modules
        ],
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )
    previous = RuntimeObservation.from_dict(observation.to_dict())
    previous.time_s = 0.0
    command = ControllerCommand(
        rotor_thrusts_n={},
        vectoring_joint_targets={},
        joint_torque_commands={},
        dock_mechanism_commands={},
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
    )
    return physical_model, graph, previous, observation, command


def test_follower_subtree_estimator_sign_and_gravity_balance() -> None:
    physical_model, graph, previous, observation, command = _case()

    estimate = FollowerSubtreeDetachWrenchEstimator().estimate(
        morphology_graph=graph,
        physical_model=physical_model,
        previous_observation=previous,
        observation=observation,
        controller_command=command,
        edge_id=0,
        follower_module_id=1,
        dt_s=0.1,
        external_contact_free=True,
    )

    assert estimate.valid is True
    assert estimate.follower_module_ids == [1]
    assert estimate.sign_convention == CUT_WRENCH_SIGN_CONVENTION
    assert estimate.force_norm_n == pytest.approx(
        physical_model.aggregate_mass_kg * 9.80665,
        rel=1.0e-6,
    )
    assert estimate.metrics["follower_module_count"] == 1.0


def test_follower_subtree_estimator_fails_closed_without_contact_free_evidence() -> None:
    physical_model, graph, previous, observation, command = _case()

    estimate = FollowerSubtreeDetachWrenchEstimator().estimate(
        morphology_graph=graph,
        physical_model=physical_model,
        previous_observation=previous,
        observation=observation,
        controller_command=command,
        edge_id=0,
        follower_module_id=1,
        dt_s=0.1,
        external_contact_free=None,
    )

    assert estimate.valid is False
    assert estimate.failure_reason == "follower_external_contact_free_evidence_missing"


def test_follower_com_to_dock_wrench_transform_has_explicit_moment_shift_sign() -> None:
    transformed = transform_follower_com_wrench_to_dock_frame(
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )

    assert transformed == pytest.approx([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])


def test_detach_unload_gate_requires_consecutive_safe_steps_and_resets() -> None:
    gate = DetachUnloadGate(DetachUnloadGateConfig(unload_dwell_steps=2))
    safe = DetachWrenchEstimate(
        edge_id=0,
        follower_module_ids=[1],
        valid=True,
        force_norm_n=0.1,
        torque_norm_nm=0.01,
    )

    first = gate.evaluate(
        estimate=safe,
        external_contact_free=True,
        parent_qp_feasible=True,
        follower_qp_feasible=True,
        relative_position_error_m=0.0,
        relative_rotation_error_rad=0.0,
        relative_linear_speed_mps=0.0,
        relative_angular_speed_radps=0.0,
    )
    second = gate.evaluate(
        estimate=safe,
        external_contact_free=True,
        parent_qp_feasible=True,
        follower_qp_feasible=True,
        relative_position_error_m=0.0,
        relative_rotation_error_rad=0.0,
        relative_linear_speed_mps=0.0,
        relative_angular_speed_radps=0.0,
    )
    failed = gate.evaluate(
        estimate=safe,
        external_contact_free=False,
        parent_qp_feasible=True,
        follower_qp_feasible=True,
        relative_position_error_m=0.0,
        relative_rotation_error_rad=0.0,
        relative_linear_speed_mps=0.0,
        relative_angular_speed_radps=0.0,
    )

    assert first.ready_to_release is False
    assert second.ready_to_release is True
    assert failed.ready_to_release is False
    assert failed.consecutive_unload_steps == 0
    assert "external_contact_not_free" in failed.failure_reasons
