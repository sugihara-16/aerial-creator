from __future__ import annotations

import math

import pytest

from amsrr.assembly.assembly_control_bridge import (
    ASSEMBLY_CONTROL_BRIDGE_CONTRACT_VERSION,
    AssemblyComponentObservation,
    AssemblyComponentSpec,
    AssemblyControlBridge,
    AssemblyControlBridgeConfig,
    AssemblyControlObservation,
    AssemblyControlRequest,
)
from amsrr.geometry.pose_math import FACE_TO_FACE_DOCK_RELATION
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import ControlGroup, DockEdge, ModuleNode, MorphologyGraph, PortNode
from amsrr.schemas.physical_model import (
    DockPortSpec,
    JointModel,
    LinkModel,
    ModuleCapabilityToken,
    PhysicalModel,
)
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL


IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def test_bridge_generates_component_v2_targets_and_exact_staging_geometry() -> None:
    bridge = _bridge()
    output = bridge.begin(_request(), _observation(time_s=0.0, axial_gap_m=0.25))

    assert output.commands.contract_version == ASSEMBLY_CONTROL_BRIDGE_CONTRACT_VERSION
    assert output.progress.phase == "staging"
    targets = {target.role: target for target in output.commands.component_targets}
    assert targets["leader"].policy_command.desired_body_pose == IDENTITY
    follower_pose = targets["follower"].policy_command.desired_body_pose
    assert follower_pose is not None
    assert follower_pose[:3] == pytest.approx((0.15, 0.0, 0.0))
    assert follower_pose[3:] == pytest.approx(FACE_TO_FACE_DOCK_RELATION[3:])
    for target in targets.values():
        command = target.policy_command
        assert command.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
        assert command.contact_tracking_bias == {}
        assert command.joint_position_targets == {
            f"module_{target.module_ids[0]}:dock_joint": 0.0
        }
        assert command.joint_velocity_targets == {
            f"module_{target.module_ids[0]}:dock_joint": 0.0
        }
        assert command.joint_torque_bias == {
            f"module_{target.module_ids[0]}:dock_joint": 0.0
        }
    assert output.commands.constraint_intent.action == "none"
    assert output.commands.constraint_intent.required_relative_pose == FACE_TO_FACE_DOCK_RELATION


def test_bridge_runs_staging_dwell_approach_fix_and_verify_state_machine() -> None:
    bridge = _bridge()
    bridge.begin(_request(), _observation(time_s=0.0, axial_gap_m=0.25))

    output = bridge.tick(_observation(time_s=0.1, axial_gap_m=0.15))
    assert output.progress.phase == "prealign_dwell"
    assert output.progress.gate_results["axial_target"]
    assert output.progress.gate_results["transverse"]
    assert output.progress.gate_results["attitude"]

    output = bridge.tick(_observation(time_s=0.45, axial_gap_m=0.15))
    assert output.progress.phase == "axial_approach"
    follower = next(target for target in output.commands.component_targets if target.role == "follower")
    assert follower.policy_command.desired_body_pose is not None
    assert follower.policy_command.desired_body_pose[0] == pytest.approx(0.15)

    output = bridge.tick(
        _observation(time_s=0.55, axial_gap_m=0.002, selected_pair_contact=True)
    )
    assert output.progress.phase == "axial_approach"
    assert not output.progress.gate_results["selected_pair_contact_dwell"]

    output = bridge.tick(
        _observation(time_s=0.66, axial_gap_m=0.002, selected_pair_contact=True)
    )
    assert output.progress.phase == "fix_ready"
    assert output.commands.constraint_intent.action == "create"
    assert output.commands.constraint_intent.required_relative_pose == FACE_TO_FACE_DOCK_RELATION

    output = bridge.tick(
        _observation(time_s=0.70, axial_gap_m=0.0, selected_pair_contact=True, constraint_present=True)
    )
    assert output.progress.phase == "verify"
    assert output.commands.constraint_intent.action == "verify"
    assert not output.progress.completed

    output = bridge.tick(
        _observation(
            time_s=0.75,
            axial_gap_m=0.0,
            selected_pair_contact=True,
            constraint_present=True,
            constraint_verified=True,
        )
    )
    assert output.progress.phase == "verify"
    assert output.progress.completed
    assert not output.progress.failed


def test_selected_contact_dwell_must_be_continuous() -> None:
    bridge = _bridge()
    bridge.begin(_request(), _observation(time_s=0.0, axial_gap_m=0.15))
    bridge.tick(_observation(time_s=0.1, axial_gap_m=0.15))
    bridge.tick(_observation(time_s=0.45, axial_gap_m=0.15))
    bridge.tick(_observation(time_s=0.50, axial_gap_m=0.002, selected_pair_contact=True))

    output = bridge.tick(_observation(time_s=0.58, axial_gap_m=0.002))
    assert output.progress.phase == "axial_approach"
    assert output.progress.selected_contact_dwell_elapsed_s == pytest.approx(0.0)

    output = bridge.tick(
        _observation(time_s=0.62, axial_gap_m=0.002, selected_pair_contact=True)
    )
    assert output.progress.phase == "axial_approach"
    output = bridge.tick(
        _observation(time_s=0.73, axial_gap_m=0.002, selected_pair_contact=True)
    )
    assert output.progress.phase == "fix_ready"


def test_bridge_splits_axial_transverse_attitude_and_twist_errors() -> None:
    bridge = _bridge()
    bridge.begin(_request(), _observation(time_s=0.0, axial_gap_m=0.25))
    observation = _observation(
        time_s=0.1,
        axial_gap_m=0.15,
        transverse_y_m=0.02,
        follower_yaw_error_rad=0.10,
        follower_linear_velocity=(0.06, 0.0, 0.0),
        follower_angular_velocity=(0.0, 0.0, 0.11),
    )

    output = bridge.tick(observation)
    error = output.progress.alignment_error
    assert error.axial_gap_m == pytest.approx(0.15)
    assert error.axial_target_error_m == pytest.approx(0.0)
    assert error.transverse_error_m == pytest.approx(0.02)
    assert error.attitude_error_rad == pytest.approx(0.10)
    assert error.relative_linear_speed_mps == pytest.approx(0.06)
    assert error.relative_angular_speed_radps == pytest.approx(0.11)
    assert output.progress.phase == "staging"
    assert not output.progress.gate_results["transverse"]
    assert not output.progress.gate_results["attitude"]
    assert not output.progress.gate_results["relative_linear_speed"]
    assert not output.progress.gate_results["relative_angular_speed"]


def test_misaligned_early_contact_fails_closed_without_constraint_intent() -> None:
    bridge = _bridge()
    bridge.begin(_request(), _observation(time_s=0.0, axial_gap_m=0.15))
    bridge.tick(_observation(time_s=0.1, axial_gap_m=0.15))
    bridge.tick(_observation(time_s=0.45, axial_gap_m=0.15))

    output = bridge.tick(
        _observation(
            time_s=0.5,
            axial_gap_m=0.04,
            transverse_y_m=0.02,
            selected_pair_contact=True,
        )
    )

    assert output.progress.phase == "safe_hold"
    assert output.progress.failed
    assert output.progress.failure_reason == "selected_pair_contact_before_fix_gate"
    assert output.commands.constraint_intent.action == "none"


def test_contact_before_approach_and_unexpected_constraint_fail_closed() -> None:
    bridge = _bridge()
    bridge.begin(_request(), _observation(time_s=0.0, axial_gap_m=0.25))
    output = bridge.tick(
        _observation(time_s=0.1, axial_gap_m=0.15, selected_pair_contact=True)
    )
    assert output.progress.phase == "safe_hold"
    assert output.progress.failure_reason == "selected_pair_contact_before_approach"

    bridge = _bridge()
    bridge.begin(_request(), _observation(time_s=0.0, axial_gap_m=0.25))
    output = bridge.tick(
        _observation(time_s=0.1, axial_gap_m=0.15, constraint_present=True)
    )
    assert output.progress.phase == "safe_hold"
    assert output.progress.failure_reason == "constraint_present_before_fix_gate"


@pytest.mark.parametrize(
    ("contact_force_n", "penetration_m", "failure_reason"),
    [
        (31.0, 0.0, "selected_pair_contact_force_exceeded"),
        (1.0, 0.003, "selected_pair_contact_penetration_exceeded"),
    ],
)
def test_excessive_selected_contact_force_or_penetration_fails_closed(
    contact_force_n: float,
    penetration_m: float,
    failure_reason: str,
) -> None:
    bridge = _bridge()
    bridge.begin(_request(), _observation(time_s=0.0, axial_gap_m=0.15))
    bridge.tick(_observation(time_s=0.1, axial_gap_m=0.15))
    bridge.tick(_observation(time_s=0.45, axial_gap_m=0.15))

    output = bridge.tick(
        _observation(
            time_s=0.5,
            axial_gap_m=0.002,
            selected_pair_contact=True,
            contact_force_n=contact_force_n,
            penetration_m=penetration_m,
        )
    )
    assert output.progress.phase == "safe_hold"
    assert output.progress.failure_reason == failure_reason
    assert output.commands.constraint_intent.action == "none"


def test_joint_correction_is_small_absolute_q_zero_offset_not_close_direction() -> None:
    bridge = _bridge()
    request = _request()
    request.follower_joint_corrections_rad = {"module_1:dock_joint": 0.04}
    output = bridge.begin(request, _observation(time_s=0.0, axial_gap_m=0.15))
    follower = next(target for target in output.commands.component_targets if target.role == "follower")
    assert follower.policy_command.joint_position_targets == {"module_1:dock_joint": 0.04}
    assert follower.policy_command.joint_torque_bias == {"module_1:dock_joint": 0.0}

    invalid = _request()
    invalid.follower_joint_corrections_rad = {"module_1:dock_joint": 0.2}
    with pytest.raises(SchemaValidationError, match="canonical-q=0 correction bound"):
        _bridge().begin(invalid, _observation(time_s=0.0, axial_gap_m=0.15))


def _bridge() -> AssemblyControlBridge:
    model = _physical_model()
    return AssemblyControlBridge(
        _morphology(),
        {0: model, 1: model},
        config=AssemblyControlBridgeConfig(
            staging_offset_m=0.15,
            prealign_dwell_s=0.30,
            max_joint_correction_rad=0.05,
        ),
    )


def _request() -> AssemblyControlRequest:
    return AssemblyControlRequest(
        step_id=7,
        leader=AssemblyComponentSpec(component_id="leader", module_ids=[0]),
        follower=AssemblyComponentSpec(component_id="follower", module_ids=[1]),
        leader_port_id=10,
        follower_port_id=11,
    )


def _observation(
    *,
    time_s: float,
    axial_gap_m: float,
    transverse_y_m: float = 0.0,
    follower_yaw_error_rad: float = 0.0,
    follower_linear_velocity=(0.0, 0.0, 0.0),
    follower_angular_velocity=(0.0, 0.0, 0.0),
    selected_pair_contact: bool = False,
    contact_force_n: float = 0.0,
    penetration_m: float = 0.0,
    constraint_present: bool = False,
    constraint_verified: bool = False,
) -> AssemblyControlObservation:
    half = 0.5 * follower_yaw_error_rad
    face_with_error = (
        axial_gap_m,
        transverse_y_m,
        0.0,
        0.0,
        0.0,
        math.cos(half),
        -math.sin(half),
    )
    return AssemblyControlObservation(
        time_s=time_s,
        components=[
            AssemblyComponentObservation(
                component_id="leader",
                module_ids=[0],
                body_pose_world=IDENTITY,
                selected_connect_pose_world=IDENTITY,
                selected_connect_linear_velocity_world=(0.0, 0.0, 0.0),
                selected_connect_angular_velocity_world=(0.0, 0.0, 0.0),
                qp_feasible=True,
            ),
            AssemblyComponentObservation(
                component_id="follower",
                module_ids=[1],
                body_pose_world=face_with_error,
                selected_connect_pose_world=face_with_error,
                selected_connect_linear_velocity_world=follower_linear_velocity,
                selected_connect_angular_velocity_world=follower_angular_velocity,
                qp_feasible=True,
            ),
        ],
        selected_pair_contact=selected_pair_contact,
        selected_pair_contact_evidence_valid=selected_pair_contact,
        selected_pair_contact_force_n=contact_force_n,
        selected_pair_penetration_m=penetration_m,
        constraint_present=constraint_present,
        constraint_verified=constraint_verified,
    )


def _morphology() -> MorphologyGraph:
    capability = ModuleCapabilityToken(
        module_type="holon",
        aggregate_mass_norm=1.0,
        aggregate_inertia_features=[],
        rotor_count=4,
        port_count=1,
        thrust_min_features=[],
        thrust_max_features=[],
        thrust_to_weight_ratio_est=2.0,
        dock_port_type_counts=[1],
        has_vectoring=True,
        has_dock_mechanism=True,
    )
    modules = [
        ModuleNode(0, "holon", IDENTITY, "leader", True, capability),
        ModuleNode(1, "holon", IDENTITY, "follower", False, capability),
    ]
    ports = [
        PortNode(10, 0, "dock", IDENTITY, "generic_dock", False, []),
        PortNode(11, 1, "dock", IDENTITY, "generic_dock", False, []),
    ]
    return MorphologyGraph(
        graph_id="order5-test",
        modules=modules,
        ports=ports,
        dock_edges=[
            DockEdge(
                edge_id=0,
                src_module_id=0,
                src_port_id=10,
                dst_module_id=1,
                dst_port_id=11,
                relative_pose_src_to_dst=FACE_TO_FACE_DOCK_RELATION,
                edge_role="structural",
                estimated_stiffness=[1.0] * 6,
                latch_state="planned",
            )
        ],
        robot_anchors=[],
        control_groups=[ControlGroup("assembly", [0, 1], "assembly")],
        base_module_id=0,
        is_closed_loop=False,
    )


def _physical_model() -> PhysicalModel:
    return PhysicalModel(
        model_id="holon-test",
        urdf_path="configurable/test.urdf",
        links=[
            LinkModel("base", None, 1.0, [1.0] * 6, (0.0, 0.0, 0.0), None, None),
            LinkModel("dock_link", "dock_joint", 0.1, [0.1] * 6, (0.0, 0.0, 0.0), None, None),
        ],
        joints=[
            JointModel(
                joint_id="dock_joint",
                joint_type="revolute",
                parent_link="base",
                child_link="dock_link",
                origin_xyz=(0.0, 0.0, 0.0),
                origin_rpy=(0.0, 0.0, 0.0),
                axis_xyz=(0.0, 0.0, 1.0),
                limit_lower=-1.0,
                limit_upper=1.0,
                effort_limit=1.0,
                velocity_limit=1.0,
            )
        ],
        rotors=[],
        dock_ports=[
            DockPortSpec(
                port_id="dock",
                parent_link="dock_link",
                local_pose=IDENTITY,
                port_type="generic_dock",
                compatible_port_types=["generic_dock"],
                mechanical_limits={"mechanism_joint_id": "dock_joint"},
            )
        ],
        collision_primitives=[],
        aggregate_mass_kg=1.1,
        aggregate_inertia_body=[1.0] * 6,
    )
