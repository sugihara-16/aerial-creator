from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from amsrr.controllers.actuator_mapping import ActuatorChannel, ActuatorMapping
from amsrr.controllers.controller_base import PayloadCoupling
from amsrr.controllers.natural_contact_joint_controller import (
    DockJointLimit,
    DockJointVector,
)
from amsrr.geometry.pose_math import pose_to_xyz_rpy
from amsrr.policies.deterministic_natural_contact_planner import (
    NaturalContactAnchorSelection,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order8 import Order8NaturalContactPhase
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    ControllerStatus,
    PolicyCommand,
)
from amsrr.schemas.runtime import ModuleRuntimeState
from amsrr.simulation.order8_contact_measurement import (
    Order8ContactPatchKinematics,
)
from amsrr.simulation.order8_isaac_runtime import (
    _SelectedMeshLocalAABB,
    _all_selected_anchor_surface_load_settled,
    _actuator_mapping_with_torque_bias_limit,
    _advance_loaded_state_rebase_settle_dwell,
    _apply_order8_seed,
    _anchor_task_linearizations,
    _anchor_tasks_from_planner_trajectory,
    _advance_pose_toward,
    _accelerate_unlatched_anchor_after_first_arrest,
    _apply_closure_direction_joint_torque_bias,
    _apply_simple_joint_velocity_command,
    _base_target_for_phase,
    _base_target_from_planner_trajectory,
    _base_twist_from_planner_trajectory,
    _alternating_reacquire_anchor_target,
    _base_hold_settled,
    _base_translation_speed_limit_for_phase,
    _centroidal_measured_joint_reference,
    _clearance_synchronized_contact_anchor_target_speed_limits_mps,
    _contact_anchor_pose_priority,
    _contact_anchor_references_with_measured_orientation,
    _contact_anchor_target_speed_limit_mps,
    _contact_anchor_target_speed_limits_mps,
    _contact_motion_entry_speed_scale,
    _contact_precenter_nominal_pose,
    _contact_centering_base_pose,
    _contact_region_pose_target,
    _contact_region_tangential_offsets_m,
    _contact_vector_telemetry_from_flat_buffers,
    _contact_force_hold_settled,
    _contact_required_motion_safety_authorized,
    _contact_force_scale_for_phase,
    _contact_pair_centering_settled,
    _contact_yield_joint_drive_gains,
    _contact_yield_tracking_profile,
    _desired_anchor_poses,
    _diagnostic_delayed_lift_bias_progress_scale,
    _diagnostic_force_stop_ready,
    _diagnostic_payload_coupling_component_flags,
    _diagnostic_payload_coupling_component_view,
    _diagnostic_prelift_controller_restore_ready,
    _damping_compensated_joint_load_nm,
    _dock_joint_armature_setting,
    _dock_joint_actuator_telemetry_entry,
    _dock_limit,
    _floor_clear_grasp_base_plan,
    _first_order_low_pass,
    _fixed_whole_structure_closure_velocity_targets,
    _global_dock_position_map,
    _gripper_object_clearance_from_body_poses,
    _gripper_object_surface_sample_clearance_from_body_poses,
    _gripper_object_surface_sample_query_from_body_poses,
    _hold_latched_joint_positions,
    _loaded_state_rebase_acceleration_bias_scale,
    _horizontal_mesh_pair_centering_correction_world,
    _hold_joint_subset_positions,
    _minimum_gripper_object_axial_overlap_from_body_poses,
    _measured_object_lift_transfer_scale,
    _lift_acceleration_bias_scale_for_phase,
    _lift_acceleration_force_bias_world,
    _mesh_aware_anchor_opening_plan,
    _mesh_aware_staging_plan,
    _module_frame_pose_twist,
    _maximum_tangential_slip_kinematics_by_link,
    _natural_contact_payload_coupling,
    _object_relative_inward_preload_pose,
    _order8_physics_dock_delta_rad,
    _post_first_arrest_centroidal_transfer_pose,
    _pose_following_object_motion,
    _parse_qclose_checkpoint_state,
    _payload_load_transfer_scale_from_external_wrench,
    _payload_feedforward_scale_for_phase,
    _prelift_relative_speed_threshold_mps,
    _project_object_rotation_state,
    _per_anchor_influential_dock_loads,
    _qclose_checkpoint_state_to_dict,
    _rebased_manipulation_base_poses,
    _rigid_point_pose_following_anchor_target,
    _joint_velocity_targets_toward_positions,
    _kit_visualizer_requested,
    _load_limited_position_preload_velocity_targets,
    _position_preload_joint_ids_by_anchor,
    _selected_anchor_surface_load_arrest_candidates,
    _selected_anchor_surface_load_settle_candidates,
    _selected_gripper_mesh_local_aabbs,
    _selected_gripper_cone_proxy_pad_specs,
    _cone_proxy_pad_surface_local_meshes,
    _selected_gripper_proxy_pad_specs,
    _schedule_contact_joint_drive_damping,
    _schedule_contact_joint_drive_impedance,
    _sequential_centroidal_transfer_limit_m,
    _sequential_latched_anchor_hold_tasks,
    _sequential_reacquire_anchor_tasks,
    _should_recenter_contact_pair,
    _spatial_jacobian_at_world_point,
    _underactuated_contact_centering_pose,
    _torque_bias_limit_with_peak_window,
    _dominant_signed_vector_axis,
    _vector_pose_local_to_world,
    _vector_world_to_pose_local,
    _advance_contact_yield_blend,
    _whole_structure_runtime_observation,
    _zero_joint_torque_bias,
)


def test_order8_physics_dock_delta_uses_only_dock_joint_indices() -> None:
    trace = {
        "joint_names_by_module": {
            "0": ["gimbal1", "yaw_dock_mech_joint1"],
            "1": ["pitch_dock_mech_joint2", "rotor1"],
        }
    }

    delta = _order8_physics_dock_delta_rad(
        trace,
        reference_by_module={"0": [0.0, 0.10], "1": [-0.20, 0.0]},
        current_by_module={"0": [9.0, 0.17], "1": [-0.24, 8.0]},
    )

    assert delta == pytest.approx(0.07)


def test_object_rotation_projection_preserves_translation_and_linear_velocity() -> None:
    projected_pose, projected_twist, deviation, angular_speed = (
        _project_object_rotation_state(
            (1.0, 2.0, 3.0, math.sin(0.1), 0.0, 0.0, math.cos(0.1)),
            (0.4, -0.5, 0.6, 0.1, -0.2, 0.3),
            locked_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )

    assert projected_pose == pytest.approx((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))
    assert projected_twist == pytest.approx((0.4, -0.5, 0.6, 0.0, 0.0, 0.0))
    assert deviation == pytest.approx(0.2)
    assert angular_speed == pytest.approx(math.sqrt(0.14))


def test_object_rotation_projection_rejects_zero_quaternion() -> None:
    with pytest.raises(SchemaValidationError):
        _project_object_rotation_state(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0,) * 6,
            locked_orientation_xyzw=(0.0, 0.0, 0.0, 0.0),
        )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (SimpleNamespace(visualizer=["kit"]), True),
        (SimpleNamespace(visualizer=("newton", "kit")), True),
        (SimpleNamespace(visualizer="kit"), True),
        (SimpleNamespace(viz="kit"), True),
        (SimpleNamespace(visualizer=["none"]), False),
        (SimpleNamespace(visualizer=None), False),
    ),
)
def test_kit_visualizer_requested_uses_app_launcher_destination(
    arguments: SimpleNamespace,
    expected: bool,
) -> None:
    assert _kit_visualizer_requested(arguments) is expected


def test_fixed_whole_structure_closure_preserves_one_shot_velocity_ratio() -> None:
    joint_ids = (
        "module_0:unused",
        "module_1:yaw_dock_mech_joint2",
        "module_2:yaw_dock_mech_joint1",
    )
    targets = _fixed_whole_structure_closure_velocity_targets(
        ordered_joint_ids=joint_ids,
        one_shot_velocity_targets_radps={
            "module_0:unused": 0.0,
            "module_1:yaw_dock_mech_joint2": 0.5,
            "module_2:yaw_dock_mech_joint1": -1.0,
        },
        maximum_speed_radps=0.02,
    )

    assert targets == {
        "module_0:unused": 0.0,
        "module_1:yaw_dock_mech_joint2": pytest.approx(0.01),
        "module_2:yaw_dock_mech_joint1": pytest.approx(-0.02),
    }


def test_fixed_whole_structure_closure_rejects_zero_one_shot_motion() -> None:
    with pytest.raises(SchemaValidationError, match="no usable joint motion"):
        _fixed_whole_structure_closure_velocity_targets(
            ordered_joint_ids=("module_1:a", "module_2:b"),
            one_shot_velocity_targets_radps={
                "module_1:a": 0.0,
                "module_2:b": 0.0,
            },
            maximum_speed_radps=0.02,
        )


def test_fixed_whole_structure_closure_holds_selected_joints_at_zero_velocity() -> None:
    joint_ids = (
        "module_0:pitch_dock_mech_joint1",
        "module_1:yaw_dock_mech_joint2",
        "module_2:yaw_dock_mech_joint1",
    )

    targets = _fixed_whole_structure_closure_velocity_targets(
        ordered_joint_ids=joint_ids,
        one_shot_velocity_targets_radps={
            "module_0:pitch_dock_mech_joint1": -1.0,
            "module_1:yaw_dock_mech_joint2": 0.5,
            "module_2:yaw_dock_mech_joint1": -0.25,
        },
        maximum_speed_radps=0.02,
        fixed_joint_ids=("module_0:pitch_dock_mech_joint1",),
    )

    assert targets == pytest.approx(
        {
            "module_0:pitch_dock_mech_joint1": 0.0,
            "module_1:yaw_dock_mech_joint2": 0.02,
            "module_2:yaw_dock_mech_joint1": -0.01,
        }
    )


def test_position_preload_uses_only_moving_influential_joints_per_side() -> None:
    joint_ids = ("shared", "fixed", "left", "right", "unrelated")

    selected = _position_preload_joint_ids_by_anchor(
        ordered_joint_ids=joint_ids,
        closure_velocity_targets_radps={
            "shared": 0.005,
            "fixed": 0.01,
            "left": 0.02,
            "right": -0.01,
            "unrelated": 0.01,
        },
        influential_joint_ids_by_anchor={
            0: ("shared", "fixed", "left"),
            1: ("shared", "fixed", "right"),
        },
        fixed_joint_ids=("fixed",),
    )

    assert selected == {
        0: ("shared", "left"),
        1: ("shared", "right"),
    }


def test_position_preload_freezes_completed_side_and_shared_joints() -> None:
    targets = _load_limited_position_preload_velocity_targets(
        ordered_joint_ids=("shared", "left", "right", "unrelated"),
        closure_velocity_targets_radps={
            "shared": 0.01,
            "left": 0.02,
            "right": -0.01,
            "unrelated": 0.02,
        },
        preload_joint_ids_by_anchor={
            0: ("shared", "left"),
            1: ("shared", "right"),
        },
        frozen_anchor_ids=(0,),
        maximum_speed_radps=0.002,
    )

    assert targets == pytest.approx(
        {
            "shared": 0.0,
            "left": 0.0,
            "right": -0.001,
            "unrelated": 0.0,
        }
    )


def test_simple_joint_velocity_command_integrates_from_previous_target() -> None:
    @dataclass(frozen=True)
    class FakeJointResult:
        policy_command: PolicyCommand

    joint_ids = ("module_1:terminal", "module_2:terminal")
    joint_vector = DockJointVector(
        joint_ids=joint_ids,
        positions_rad=(0.10, -0.20),
        velocities_radps=(0.0, 0.0),
        neutral_positions_rad=(0.0, 0.0),
        limits=(
            DockJointLimit(-1.0, 1.0, 0.05, 4.0),
            DockJointLimit(-0.205, 1.0, 0.05, 4.0),
        ),
    )
    result = FakeJointResult(
        PolicyCommand(
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets={
                "module_1:terminal": 0.5,
                "module_2:terminal": 0.5,
            },
            joint_velocity_targets={
                "module_1:terminal": 0.0,
                "module_2:terminal": 0.0,
            },
            joint_torque_bias={
                "module_1:terminal": 2.0,
                "module_2:terminal": -2.0,
            },
        )
    )

    commanded = _apply_simple_joint_velocity_command(
        result,
        joint_vector,
        velocity_targets_radps={
            "module_1:terminal": 0.02,
            "module_2:terminal": -0.10,
        },
        previous_position_targets_rad={
            "module_1:terminal": 0.30,
            "module_2:terminal": -0.20,
        },
        dt_s=0.1,
        zero_torque_bias=True,
    )

    assert commanded.policy_command.joint_position_targets == pytest.approx(
        {
            "module_1:terminal": 0.302,
            "module_2:terminal": -0.205,
        }
    )
    assert commanded.policy_command.joint_velocity_targets == pytest.approx(
        {
            "module_1:terminal": 0.02,
            "module_2:terminal": -0.05,
        }
    )
    assert commanded.policy_command.joint_torque_bias == {
        "module_1:terminal": 0.0,
        "module_2:terminal": 0.0,
    }


def test_zero_joint_torque_bias_clears_policy_mapping_and_diagnostics() -> None:
    @dataclass(frozen=True)
    class FakeTorqueMapping:
        unclipped_joint_torque_bias: dict[str, float]
        joint_torque_bias: dict[str, float]
        clipped_joint_ids: tuple[str, ...]

    @dataclass(frozen=True)
    class FakeDiagnostics:
        torque_clipped_joint_ids: tuple[str, ...]

    @dataclass(frozen=True)
    class FakeResult:
        policy_command: PolicyCommand
        torque_mapping: FakeTorqueMapping
        diagnostics: FakeDiagnostics

    joint_ids = ("module_1:left", "module_2:right")
    joint_vector = DockJointVector(
        joint_ids=joint_ids,
        positions_rad=(0.1, -0.1),
        velocities_radps=(0.0, 0.0),
        neutral_positions_rad=(0.0, 0.0),
        limits=(
            DockJointLimit(-1.0, 1.0, 0.1, 4.1),
            DockJointLimit(-1.0, 1.0, 0.1, 4.1),
        ),
    )
    result = FakeResult(
        policy_command=PolicyCommand(
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets={joint_ids[0]: 0.2, joint_ids[1]: -0.2},
            joint_velocity_targets={joint_ids[0]: 0.01, joint_ids[1]: -0.01},
            joint_torque_bias={joint_ids[0]: 1.0, joint_ids[1]: -1.0},
        ),
        torque_mapping=FakeTorqueMapping(
            unclipped_joint_torque_bias={joint_ids[0]: 2.0, joint_ids[1]: -2.0},
            joint_torque_bias={joint_ids[0]: 1.0, joint_ids[1]: -1.0},
            clipped_joint_ids=joint_ids,
        ),
        diagnostics=FakeDiagnostics(torque_clipped_joint_ids=joint_ids),
    )

    zeroed = _zero_joint_torque_bias(result, joint_vector)

    assert zeroed.policy_command.joint_position_targets == (
        result.policy_command.joint_position_targets
    )
    assert zeroed.policy_command.joint_velocity_targets == (
        result.policy_command.joint_velocity_targets
    )
    assert zeroed.policy_command.joint_torque_bias == {
        joint_id: 0.0 for joint_id in joint_ids
    }
    assert zeroed.torque_mapping.unclipped_joint_torque_bias == {
        joint_id: 0.0 for joint_id in joint_ids
    }
    assert zeroed.torque_mapping.joint_torque_bias == {
        joint_id: 0.0 for joint_id in joint_ids
    }
    assert zeroed.torque_mapping.clipped_joint_ids == ()
    assert zeroed.diagnostics.torque_clipped_joint_ids == ()


def test_directional_joint_torque_bias_uses_closure_sign_for_selected_joints() -> None:
    @dataclass(frozen=True)
    class FakeTorqueMapping:
        unclipped_joint_torque_bias: dict[str, float]
        joint_torque_bias: dict[str, float]
        clipped_joint_ids: tuple[str, ...]

    @dataclass(frozen=True)
    class FakeDiagnostics:
        torque_clipped_joint_ids: tuple[str, ...]

    @dataclass(frozen=True)
    class FakeResult:
        policy_command: PolicyCommand
        torque_mapping: FakeTorqueMapping
        diagnostics: FakeDiagnostics

    joint_ids = ("left", "right", "pitch_hold")
    joint_vector = DockJointVector(
        joint_ids=joint_ids,
        positions_rad=(0.0, 0.0, 0.0),
        velocities_radps=(0.0, 0.0, 0.0),
        neutral_positions_rad=(0.0, 0.0, 0.0),
        limits=tuple(DockJointLimit(-1.0, 1.0, 0.1, 1.3) for _ in joint_ids),
    )
    zeros = {joint_id: 0.0 for joint_id in joint_ids}
    result = FakeResult(
        policy_command=PolicyCommand(
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets=dict(zeros),
            joint_velocity_targets=dict(zeros),
            joint_torque_bias=dict(zeros),
        ),
        torque_mapping=FakeTorqueMapping(dict(zeros), dict(zeros), ("left",)),
        diagnostics=FakeDiagnostics(("left",)),
    )

    biased = _apply_closure_direction_joint_torque_bias(
        result,
        joint_vector,
        closure_velocity_targets_radps={
            "left": 0.02,
            "right": -0.03,
            "pitch_hold": 0.0,
        },
        selected_joint_ids=("left", "right"),
        magnitude_nm=0.5,
    )

    expected = {"left": 0.5, "right": -0.5, "pitch_hold": 0.0}
    assert biased.policy_command.joint_torque_bias == expected
    assert biased.torque_mapping.unclipped_joint_torque_bias == expected
    assert biased.torque_mapping.joint_torque_bias == expected
    assert biased.torque_mapping.clipped_joint_ids == ()
    assert biased.diagnostics.torque_clipped_joint_ids == ()


@pytest.mark.parametrize("magnitude_nm", (0.0, 1.31))
def test_directional_joint_torque_bias_rejects_invalid_magnitude(
    magnitude_nm: float,
) -> None:
    @dataclass(frozen=True)
    class FakeTorqueMapping:
        unclipped_joint_torque_bias: dict[str, float]
        joint_torque_bias: dict[str, float]
        clipped_joint_ids: tuple[str, ...]

    @dataclass(frozen=True)
    class FakeDiagnostics:
        torque_clipped_joint_ids: tuple[str, ...]

    @dataclass(frozen=True)
    class FakeResult:
        policy_command: PolicyCommand
        torque_mapping: FakeTorqueMapping
        diagnostics: FakeDiagnostics

    joint_vector = DockJointVector(
        joint_ids=("joint",),
        positions_rad=(0.0,),
        velocities_radps=(0.0,),
        neutral_positions_rad=(0.0,),
        limits=(DockJointLimit(-1.0, 1.0, 0.1, 1.3),),
    )
    zeros = {"joint": 0.0}
    result = FakeResult(
        policy_command=PolicyCommand(
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets=dict(zeros),
            joint_velocity_targets=dict(zeros),
            joint_torque_bias=dict(zeros),
        ),
        torque_mapping=FakeTorqueMapping(dict(zeros), dict(zeros), ()),
        diagnostics=FakeDiagnostics(()),
    )

    with pytest.raises(SchemaValidationError):
        _apply_closure_direction_joint_torque_bias(
            result,
            joint_vector,
            closure_velocity_targets_radps={"joint": 0.02},
            selected_joint_ids=("joint",),
            magnitude_nm=magnitude_nm,
        )


def test_joint_subset_position_hold_overrides_motion_and_torque() -> None:
    @dataclass(frozen=True)
    class FakeJointResult:
        policy_command: PolicyCommand

    joint_ids = (
        "module_0:pitch_dock_mech_joint1",
        "module_0:yaw_dock_mech_joint1",
    )
    joint_vector = DockJointVector(
        joint_ids=joint_ids,
        positions_rad=(0.10, -0.20),
        velocities_radps=(0.0, 0.0),
        neutral_positions_rad=(0.0, 0.0),
        limits=(
            DockJointLimit(-1.0, 1.0, 0.05, 4.0),
            DockJointLimit(-1.0, 1.0, 0.05, 4.0),
        ),
    )
    result = FakeJointResult(
        PolicyCommand(
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets={
                joint_ids[0]: 0.50,
                joint_ids[1]: -0.30,
            },
            joint_velocity_targets={
                joint_ids[0]: 0.02,
                joint_ids[1]: -0.02,
            },
            joint_torque_bias={
                joint_ids[0]: 1.0,
                joint_ids[1]: -1.0,
            },
        )
    )

    held = _hold_joint_subset_positions(
        result,
        joint_vector,
        position_targets_rad={joint_ids[0]: 0.12},
    )

    assert held.policy_command.joint_position_targets == pytest.approx(
        {joint_ids[0]: 0.12, joint_ids[1]: -0.30}
    )
    assert held.policy_command.joint_velocity_targets == pytest.approx(
        {joint_ids[0]: 0.0, joint_ids[1]: -0.02}
    )
    assert held.policy_command.joint_torque_bias == pytest.approx(
        {joint_ids[0]: 0.0, joint_ids[1]: -1.0}
    )


def test_direct_release_velocity_returns_toward_measured_open_q() -> None:
    joint_vector = DockJointVector(
        joint_ids=("module_1:terminal", "module_2:terminal"),
        positions_rad=(0.20, -0.20),
        velocities_radps=(0.0, 0.0),
        neutral_positions_rad=(0.0, 0.0),
        limits=(
            DockJointLimit(-1.0, 1.0, 0.10, 4.0),
            DockJointLimit(-1.0, 1.0, 0.10, 4.0),
        ),
    )
    targets = _joint_velocity_targets_toward_positions(
        joint_vector,
        target_positions_rad={
            "module_1:terminal": 0.10,
            "module_2:terminal": -0.199,
        },
        maximum_speed_radps=0.02,
        dt_s=0.1,
    )

    assert targets == pytest.approx(
        {
            "module_1:terminal": -0.02,
            "module_2:terminal": 0.01,
        }
    )


def test_diagnostic_partial_force_ramp_can_stop_without_grasp_dwell() -> None:
    assert _diagnostic_force_stop_ready(
        contact_configuration_latched=True,
        contact_force_scale=0.4,
        stop_force_scale=0.4,
        grasp_acquired=False,
    )


def test_diagnostic_full_force_ramp_waits_for_stable_grasp() -> None:
    assert not _diagnostic_force_stop_ready(
        contact_configuration_latched=True,
        contact_force_scale=1.0,
        stop_force_scale=1.0,
        grasp_acquired=False,
    )
    assert _diagnostic_force_stop_ready(
        contact_configuration_latched=True,
        contact_force_scale=1.0,
        stop_force_scale=1.0,
        grasp_acquired=True,
    )


def test_downstream_waypoints_preserve_measured_grasp_orientation() -> None:
    measured_grasp_pose = (
        1.0,
        2.0,
        3.0,
        0.1,
        -0.2,
        0.3,
        math.sqrt(0.86),
    )

    lift, transport, place, retreat = _rebased_manipulation_base_poses(
        measured_grasp_pose,
        transport_distance_m=0.4,
    )

    assert lift[:3] == pytest.approx((1.0, 2.0, 3.15))
    assert transport[:3] == pytest.approx((1.4, 2.0, 3.15))
    assert place[:3] == pytest.approx((1.4, 2.0, 3.0))
    assert retreat[:3] == pytest.approx((1.3, 2.0, 3.2))
    assert lift[3:] == measured_grasp_pose[3:]
    assert transport[3:] == measured_grasp_pose[3:]
    assert place[3:] == measured_grasp_pose[3:]
    assert retreat[3:] == measured_grasp_pose[3:]


def test_object_relative_inward_preload_pose_tracks_object_frame() -> None:
    anchor_object = _object_relative_inward_preload_pose(
        anchor_pose_world=(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
        object_pose_world=(0.5, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
        inward_normal_object=(-2.0, 0.0, 0.0),
        preload_distance_m=0.004,
    )
    assert anchor_object == pytest.approx(
        (0.496, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    )


def test_spatial_jacobian_shift_maps_rotation_to_surface_point_velocity() -> None:
    shifted = _spatial_jacobian_at_world_point(
        (
            (0.0,),
            (0.0,),
            (0.0,),
            (0.0,),
            (0.0,),
            (1.0,),
        ),
        origin_world=(0.0, 0.0, 0.0),
        point_world=(1.0, 0.0, 0.0),
    )
    expected = (
        (0.0,),
        (1.0,),
        (0.0,),
        (0.0,),
        (0.0,),
        (1.0,),
    )
    assert len(shifted) == len(expected)
    for actual_row, expected_row in zip(shifted, expected, strict=True):
        assert actual_row == pytest.approx(expected_row)


def test_rigid_surface_point_follows_anchor_translation_and_rotation() -> None:
    desired = _rigid_point_pose_following_anchor_target(
        current_anchor_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        current_point_world=(1.0, 0.0, 0.0),
        desired_anchor_pose_world=(
            2.0,
            3.0,
            4.0,
            0.0,
            0.0,
            math.sqrt(0.5),
            math.sqrt(0.5),
        ),
    )
    assert desired[:3] == pytest.approx((2.0, 4.0, 4.0))


def test_contact_yield_blend_ramps_down_and_restores_without_overshoot() -> None:
    blend = _advance_contact_yield_blend(
        0.0,
        yield_requested=True,
        dt_s=0.10,
        ramp_down_s=0.25,
        ramp_up_s=0.50,
    )
    assert blend == pytest.approx(0.4)
    blend = _advance_contact_yield_blend(
        blend,
        yield_requested=True,
        dt_s=1.0,
        ramp_down_s=0.25,
        ramp_up_s=0.50,
    )
    assert blend == 1.0
    blend = _advance_contact_yield_blend(
        blend,
        yield_requested=False,
        dt_s=0.20,
        ramp_down_s=0.25,
        ramp_up_s=0.50,
    )
    assert blend == pytest.approx(0.6)


def test_contact_yield_tracking_profile_preserves_full_centroidal_tracking() -> None:
    profile = _contact_yield_tracking_profile(
        0.75,
        integrator_decay_rate_per_s=12.0,
    )

    assert profile.proportional_gain_scale == 1.0
    assert profile.integral_gain_scale == 1.0
    assert profile.derivative_gain_scale == 1.0
    assert profile.integrator_accumulation_scale == 1.0
    assert profile.integrator_decay_rate_per_s == 0.0


def test_contact_load_subtracts_virtual_drive_damping_with_sign() -> None:
    assert _damping_compensated_joint_load_nm(
        applied_torque_nm=0.16,
        estimated_damping_drive_torque_nm=0.15,
    ) == pytest.approx(0.01)
    assert _damping_compensated_joint_load_nm(
        applied_torque_nm=-0.20,
        estimated_damping_drive_torque_nm=-0.05,
    ) == pytest.approx(0.15)

    with pytest.raises(SchemaValidationError, match="must be finite"):
        _damping_compensated_joint_load_nm(
            applied_torque_nm=math.nan,
            estimated_damping_drive_torque_nm=0.0,
        )


def test_contact_yield_joint_drive_gains_blend_only_simulator_impedance() -> None:
    assert _contact_yield_joint_drive_gains(
        0.0,
        nominal_stiffness_nm_per_rad=200.0,
        nominal_damping_nms_per_rad=5.0,
        yield_stiffness_scale=0.25,
        yield_damping_nms_per_rad=8.0,
    ) == pytest.approx((200.0, 5.0))
    assert _contact_yield_joint_drive_gains(
        0.4,
        nominal_stiffness_nm_per_rad=200.0,
        nominal_damping_nms_per_rad=5.0,
        yield_stiffness_scale=0.25,
        yield_damping_nms_per_rad=8.0,
    ) == pytest.approx((140.0, 6.2))
    assert _contact_yield_joint_drive_gains(
        1.0,
        nominal_stiffness_nm_per_rad=200.0,
        nominal_damping_nms_per_rad=5.0,
        yield_stiffness_scale=0.25,
        yield_damping_nms_per_rad=8.0,
    ) == pytest.approx((50.0, 8.0))


def test_contact_yield_helpers_fail_closed_on_invalid_input() -> None:
    with pytest.raises(SchemaValidationError, match="must remain in"):
        _advance_contact_yield_blend(
            1.1,
            yield_requested=True,
            dt_s=0.01,
            ramp_down_s=0.25,
            ramp_up_s=0.50,
        )
    with pytest.raises(SchemaValidationError, match="inputs are invalid"):
        _contact_yield_tracking_profile(
            -0.1,
            integrator_decay_rate_per_s=12.0,
        )
    with pytest.raises(SchemaValidationError, match="must not exceed one"):
        _contact_yield_joint_drive_gains(
            1.0,
            nominal_stiffness_nm_per_rad=200.0,
            nominal_damping_nms_per_rad=5.0,
            yield_stiffness_scale=1.01,
            yield_damping_nms_per_rad=8.0,
        )


def test_contact_vector_telemetry_keeps_normal_and_friction_vectors_separate() -> None:
    telemetry = _contact_vector_telemetry_from_flat_buffers(
        body_identity=["module_1:dock", "module_2:dock", "module_0:body"],
        selected_link_ids={"module_1:dock", "module_2:dock"},
        contact_counts=[2, 1, 0],
        contact_starts=[0, 2, 3],
        normal_force_magnitudes_n=[2.0, 3.0, 4.0],
        contact_normals_world=[
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        ],
        contact_points_world=[
            (1.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
        ],
        friction_counts=[1, 2, 0],
        friction_starts=[0, 1, 3],
        friction_forces_world=[
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 3.0),
        ],
        contact_force_matrix_world=[
            (2.0, 3.0, 1.0),
            (-4.0, 0.0, 5.0),
            (0.0, 0.0, 0.0),
        ],
    )

    assert telemetry.valid is True
    assert telemetry.normal_force_world_by_link == {
        "module_1:dock": pytest.approx((2.0, 3.0, 0.0)),
        "module_2:dock": pytest.approx((-4.0, 0.0, 0.0)),
    }
    assert telemetry.normal_force_application_point_world_by_link == {
        "module_1:dock": pytest.approx((2.2, 0.0, 0.0)),
        "module_2:dock": pytest.approx((5.0, 0.0, 0.0)),
    }
    assert telemetry.friction_force_world_by_link == {
        "module_1:dock": pytest.approx((0.0, 0.0, 1.0)),
        "module_2:dock": pytest.approx((0.0, 0.0, 5.0)),
    }
    assert telemetry.contact_force_matrix_world_by_link == {
        "module_1:dock": pytest.approx((2.0, 3.0, 1.0)),
        "module_2:dock": pytest.approx((-4.0, 0.0, 5.0)),
    }
    assert telemetry.tangential_slip_velocity_world_by_link == {
        "module_1:dock": (0.0, 0.0, 0.0),
        "module_2:dock": (0.0, 0.0, 0.0),
    }


def test_maximum_tangential_slip_kinematics_matches_scalar_patch_choice() -> None:
    def patch(
        patch_id: str,
        link_id: str,
        velocity: tuple[float, float, float],
    ) -> Order8ContactPatchKinematics:
        return Order8ContactPatchKinematics(
            patch_id=patch_id,
            robot_link_id=link_id,
            contact_point_world=(0.0, 0.0, 0.0),
            contact_normal_world=(1.0, 0.0, 0.0),
            body_contact_velocity_world_mps=velocity,
            object_contact_velocity_world_mps=(0.0, 0.0, 0.0),
            relative_velocity_world_mps=velocity,
            tangential_velocity_world_mps=velocity,
        )

    first_tie = patch("first", "module_1:dock", (0.0, 0.03, 0.0))
    second_tie = patch("second", "module_1:dock", (0.0, -0.03, 0.0))
    maximum = patch("maximum", "module_2:dock", (0.0, 0.0, -0.04))
    ignored = patch("ignored", "module_0:body", (0.0, 1.0, 0.0))

    selected = _maximum_tangential_slip_kinematics_by_link(
        [first_tie, second_tie, maximum, ignored],
        selected_link_ids={"module_1:dock", "module_2:dock"},
    )

    assert selected == {
        "module_1:dock": first_tie,
        "module_2:dock": maximum,
    }


def test_slip_vector_frame_and_signed_dominant_axis_are_deterministic() -> None:
    half_sqrt = math.sqrt(0.5)
    object_pose_world = (0.0, 0.0, 0.0, 0.0, 0.0, half_sqrt, half_sqrt)

    vector_object = _vector_world_to_pose_local(
        object_pose_world,
        (0.0, 0.2, 0.0),
    )

    assert vector_object == pytest.approx((0.2, 0.0, 0.0), abs=1.0e-12)
    assert _vector_pose_local_to_world(
        object_pose_world,
        vector_object,
    ) == pytest.approx((0.0, 0.2, 0.0), abs=1.0e-12)
    assert _dominant_signed_vector_axis(vector_object) == "+x"
    assert _dominant_signed_vector_axis((0.001, -0.004, 0.003)) == "-y"
    assert _dominant_signed_vector_axis((0.0, 0.0, 0.0)) == "stationary"


def test_contact_vector_telemetry_fails_closed_on_bad_buffer_range() -> None:
    telemetry = _contact_vector_telemetry_from_flat_buffers(
        body_identity=["module_1:dock", "module_2:dock"],
        selected_link_ids={"module_1:dock", "module_2:dock"},
        contact_counts=[2, 1],
        contact_starts=[0, 2],
        normal_force_magnitudes_n=[1.0, 1.0],
        contact_normals_world=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        contact_points_world=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        friction_counts=[0, 0],
        friction_starts=[0, 0],
        friction_forces_world=[],
        contact_force_matrix_world=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    )

    assert telemetry.valid is False


@pytest.mark.parametrize(
    "phase",
    [
        Order8NaturalContactPhase.CONTACT_ACQUISITION,
        Order8NaturalContactPhase.LIFT,
        Order8NaturalContactPhase.TRANSPORT,
        Order8NaturalContactPhase.PLACE,
    ],
)
def test_contact_required_phases_use_contact_base_speed_limit(
    phase: Order8NaturalContactPhase,
) -> None:
    assert _base_translation_speed_limit_for_phase(
        phase,
        free_motion_limit_mps=0.10,
        maintained_contact_limit_mps=0.01,
    ) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "phase",
    [
        Order8NaturalContactPhase.APPROACH,
        Order8NaturalContactPhase.RELEASE,
        Order8NaturalContactPhase.RETREAT,
        Order8NaturalContactPhase.SETTLE,
    ],
)
def test_non_contact_motion_phases_use_free_base_speed_limit(
    phase: Order8NaturalContactPhase,
) -> None:
    assert _base_translation_speed_limit_for_phase(
        phase,
        free_motion_limit_mps=0.10,
        maintained_contact_limit_mps=0.01,
    ) == pytest.approx(0.10)


@pytest.mark.parametrize(
    (
        "phase",
        "elapsed_s",
        "estimated_lift_scale",
        "measured_lift_scale",
        "previous_scale",
        "dt_s",
        "lift_off_confirmed",
        "expected_scale",
    ),
    [
        (Order8NaturalContactPhase.CONTACT_ACQUISITION, 0.0, None, None, 0.0, None, False, 0.0),
        (Order8NaturalContactPhase.LIFT, 0.0, 0.0, 0.0, 0.0, 0.025, False, 0.0),
        (Order8NaturalContactPhase.LIFT, 0.1, 0.0, 0.0, 0.0, 0.025, False, 0.1),
        (Order8NaturalContactPhase.LIFT, 0.2, 0.5, 0.0, 0.4, 0.025, False, 0.5),
        (Order8NaturalContactPhase.LIFT, 0.3, 0.1, 0.6, 0.5, 0.025, False, 0.6),
        (Order8NaturalContactPhase.LIFT, 0.4, 0.2, 0.2, 0.9, 0.025, True, 1.0),
        (Order8NaturalContactPhase.TRANSPORT, 0.0, None, None, 0.0, None, False, 1.0),
        (Order8NaturalContactPhase.PLACE, 0.0, None, None, 0.0, None, False, 1.0),
        (Order8NaturalContactPhase.RELEASE, 0.0, None, None, 0.0, None, False, 1.0),
        (Order8NaturalContactPhase.RELEASE, 0.125, None, None, 0.0, None, False, 0.5),
        (Order8NaturalContactPhase.RELEASE, 0.25, None, None, 0.0, None, False, 0.0),
        (Order8NaturalContactPhase.RETREAT, 0.0, None, None, 0.0, None, False, 0.0),
    ],
)
def test_payload_feedforward_scale_follows_load_with_slew_and_release_ramp(
    phase: Order8NaturalContactPhase,
    elapsed_s: float,
    estimated_lift_scale: float | None,
    measured_lift_scale: float | None,
    previous_scale: float,
    dt_s: float | None,
    lift_off_confirmed: bool,
    expected_scale: float,
) -> None:
    assert _payload_feedforward_scale_for_phase(
        phase,
        phase_elapsed_s=elapsed_s,
        transition_duration_s=0.25,
        estimated_lift_transfer_scale=estimated_lift_scale,
        measured_lift_transfer_scale=measured_lift_scale,
        previous_scale=previous_scale,
        dt_s=dt_s,
        lift_off_confirmed=lift_off_confirmed,
    ) == pytest.approx(expected_scale)


@pytest.mark.parametrize(
    (
        "phase",
        "commanded_progress",
        "lift_off_elapsed_s",
        "lift_off_scale",
        "expected_scale",
    ),
    [
        (Order8NaturalContactPhase.APPROACH, 0.5, None, None, 0.0),
        (Order8NaturalContactPhase.LIFT, 0.0, None, None, 0.0),
        (Order8NaturalContactPhase.LIFT, 0.4, None, None, 0.4),
        (Order8NaturalContactPhase.LIFT, 0.8, 0.0, 0.4, 0.4),
        (Order8NaturalContactPhase.LIFT, 1.0, 0.125, 0.4, 0.3375),
        (Order8NaturalContactPhase.LIFT, 1.0, 0.25, 0.4, 0.2),
        (Order8NaturalContactPhase.LIFT, 1.0, 0.50, 0.4, 0.0),
        (Order8NaturalContactPhase.TRANSPORT, 0.0, 0.25, 0.4, 0.0),
    ],
)
def test_lift_acceleration_bias_follows_progress_then_tapers_after_lift_off(
    phase: Order8NaturalContactPhase,
    commanded_progress: float,
    lift_off_elapsed_s: float | None,
    lift_off_scale: float | None,
    expected_scale: float,
) -> None:
    assert _lift_acceleration_bias_scale_for_phase(
        phase,
        commanded_lift_progress_scale=commanded_progress,
        lift_off_elapsed_s=lift_off_elapsed_s,
        lift_off_scale=lift_off_scale,
        removal_duration_s=0.5,
    ) == pytest.approx(expected_scale)


def test_lift_acceleration_bias_is_known_payload_inertial_force_only() -> None:
    force_world = _lift_acceleration_force_bias_world(
        payload_mass_kg=2.0,
        lift_payload_acceleration_mps2=1.5,
        scale=0.25,
    )

    assert force_world == pytest.approx((0.0, 0.0, 0.75))
    half_sqrt = math.sqrt(0.5)
    force_body = _vector_world_to_pose_local(
        (0.0, 0.0, 0.0, 0.0, half_sqrt, 0.0, half_sqrt),
        force_world,
    )
    assert _vector_pose_local_to_world(
        (0.0, 0.0, 0.0, 0.0, half_sqrt, 0.0, half_sqrt),
        force_body,
    ) == pytest.approx(force_world, abs=1.0e-12)


def test_separated_lift_waits_for_complete_restore_and_slow_base() -> None:
    ready = {
        "enabled": True,
        "grasp_pose_rebased": True,
        "centroidal_yield_blend": 0.0,
        "joint_drive_yield_blend": 0.0,
        "admittance_active": False,
        "base_linear_speed_mps": 0.019,
        "base_speed_limit_mps": 0.020,
    }
    assert _diagnostic_prelift_controller_restore_ready(**ready)

    for change in (
        {"grasp_pose_rebased": False},
        {"centroidal_yield_blend": 0.01},
        {"joint_drive_yield_blend": 0.01},
        {"admittance_active": True},
        {"base_linear_speed_mps": 0.021},
    ):
        inputs = {**ready, **change}
        assert not _diagnostic_prelift_controller_restore_ready(**inputs)

    assert _diagnostic_prelift_controller_restore_ready(
        **{**ready, "enabled": False, "grasp_pose_rebased": False}
    )


@pytest.mark.parametrize(
    ("elapsed_s", "expected"),
    [
        (0.0, 0.0),
        (1.25, 0.0),
        (1.50, 0.25),
        (2.25, 1.0),
    ],
)
def test_separated_lift_delays_only_extra_bias_progress(
    elapsed_s: float,
    expected: float,
) -> None:
    assert _diagnostic_delayed_lift_bias_progress_scale(
        Order8NaturalContactPhase.LIFT,
        enabled=True,
        phase_elapsed_s=elapsed_s,
        bias_delay_s=1.25,
        transition_duration_s=1.0,
        normal_commanded_progress_scale=min(1.0, elapsed_s),
    ) == pytest.approx(expected)


def test_disabled_lift_separation_preserves_normal_bias_progress() -> None:
    assert _diagnostic_delayed_lift_bias_progress_scale(
        Order8NaturalContactPhase.LIFT,
        enabled=False,
        phase_elapsed_s=0.1,
        bias_delay_s=1.25,
        transition_duration_s=1.0,
        normal_commanded_progress_scale=0.1,
    ) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"commanded_lift_progress_scale": -0.1},
        {"lift_off_elapsed_s": 0.0, "lift_off_scale": None},
        {"lift_off_elapsed_s": None, "lift_off_scale": 0.5},
        {"lift_off_elapsed_s": -0.1, "lift_off_scale": 0.5},
        {"lift_off_elapsed_s": 0.1, "lift_off_scale": 1.1},
        {"removal_duration_s": 0.0},
    ],
)
def test_lift_acceleration_bias_rejects_invalid_schedule(kwargs: dict) -> None:
    arguments = {
        "commanded_lift_progress_scale": 0.5,
        "lift_off_elapsed_s": None,
        "lift_off_scale": None,
        "removal_duration_s": 0.5,
    }
    arguments.update(kwargs)
    with pytest.raises(SchemaValidationError, match="lift-acceleration"):
        _lift_acceleration_bias_scale_for_phase(
            Order8NaturalContactPhase.LIFT,
            **arguments,
        )


def test_payload_load_transfer_scale_uses_downward_external_force_delta() -> None:
    scale, force_z, load_n = _payload_load_transfer_scale_from_external_wrench(
        external_wrench_body=(0.0, 0.0, -4.903325, 0.0, 0.0, 0.0),
        body_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        lift_start_external_force_world_z_n=0.0,
        payload_mass_kg=1.0,
        gravity_mps2=9.80665,
    )

    assert force_z == pytest.approx(-4.903325)
    assert load_n == pytest.approx(4.903325)
    assert scale == pytest.approx(0.5)


def test_payload_load_transfer_scale_rotates_body_force_and_clips() -> None:
    half_sqrt = math.sqrt(0.5)
    scale, force_z, load_n = _payload_load_transfer_scale_from_external_wrench(
        # A +X body force becomes -Z in world after +90 deg rotation about Y.
        external_wrench_body=(20.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        body_pose_world=(0.0, 0.0, 0.0, 0.0, half_sqrt, 0.0, half_sqrt),
        lift_start_external_force_world_z_n=30.0,
        payload_mass_kg=1.0,
        gravity_mps2=9.80665,
    )

    assert force_z == pytest.approx(-20.0, abs=1.0e-12)
    assert load_n == pytest.approx(50.0)
    assert scale == pytest.approx(1.0)


def test_measured_object_lift_transfer_scale_uses_qclose_com_rise() -> None:
    qclose = (1.0, 2.0, 0.075, 0.0, 0.0, 0.0, 1.0)

    assert _measured_object_lift_transfer_scale(
        qclose_object_pose=qclose,
        current_object_pose=(1.0, 2.0, 0.080, 0.0, 0.0, 0.0, 1.0),
        transfer_distance_m=0.010,
    ) == pytest.approx(0.5)
    assert _measured_object_lift_transfer_scale(
        qclose_object_pose=qclose,
        current_object_pose=(1.0, 2.0, 0.070, 0.0, 0.0, 0.0, 1.0),
        transfer_distance_m=0.010,
    ) == pytest.approx(0.0)
    assert _measured_object_lift_transfer_scale(
        qclose_object_pose=qclose,
        current_object_pose=(1.0, 2.0, 0.095, 0.0, 0.0, 0.0, 1.0),
        transfer_distance_m=0.010,
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("elapsed_s", "duration_s"),
    [(-0.001, 0.25), (math.nan, 0.25), (0.0, 0.0), (0.0, math.inf)],
)
def test_payload_feedforward_scale_rejects_invalid_timing(
    elapsed_s: float,
    duration_s: float,
) -> None:
    with pytest.raises(SchemaValidationError, match="payload feed-forward"):
        _payload_feedforward_scale_for_phase(
            Order8NaturalContactPhase.LIFT,
            phase_elapsed_s=elapsed_s,
            transition_duration_s=duration_s,
        )


def test_natural_contact_payload_coupling_uses_measured_com_and_cuboid_inertia() -> (
    None
):
    coupling = _natural_contact_payload_coupling(
        control_body_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        object_com_pose_world=(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
        object_mass_kg=1.0,
        object_size_m=(0.3, 0.4, 0.15),
        load_transfer_scale=0.5,
        contact_model="natural_contact_grasp_v1",
    )

    assert coupling is not None
    assert coupling.mass_kg == pytest.approx(0.5)
    assert coupling.com_offset_body == pytest.approx((1.0, 2.0, 3.0))
    assert coupling.inertia_body == pytest.approx(
        [
            0.5 * (0.4**2 + 0.15**2) / 12.0,
            0.0,
            0.0,
            0.5 * (0.3**2 + 0.15**2) / 12.0,
            0.0,
            0.5 * (0.3**2 + 0.4**2) / 12.0,
        ]
    )
    assert coupling.contact_model == "natural_contact_grasp_v1"
    assert coupling.coupling_mode == (
        "natural_contact_verified_grasp_ramped_payload_v2"
    )


@pytest.mark.parametrize(
    ("mode", "expected_offset", "expected_inertia", "expected_flags"),
    [
        (
            "full",
            (0.4, -0.1, 0.05),
            [0.01, 0.0, 0.0, 0.02, 0.0, 0.03],
            {
                "translational_force": True,
                "com_offset_moment": True,
                "rotational_inertia": True,
            },
        ),
        (
            "translational_force_only",
            (0.0, 0.0, 0.0),
            [0.0] * 6,
            {
                "translational_force": True,
                "com_offset_moment": False,
                "rotational_inertia": False,
            },
        ),
        (
            "translational_force_and_com_offset_moment",
            (0.4, -0.1, 0.05),
            [0.0] * 6,
            {
                "translational_force": True,
                "com_offset_moment": True,
                "rotational_inertia": False,
            },
        ),
    ],
)
def test_diagnostic_payload_component_view_preserves_only_selected_terms(
    mode: str,
    expected_offset: tuple[float, float, float],
    expected_inertia: list[float],
    expected_flags: dict[str, bool],
) -> None:
    full = PayloadCoupling(
        payload_id="box_01",
        contact_model="natural_contact_grasp_v1",
        mass_kg=0.75,
        inertia_body=[0.01, 0.0, 0.0, 0.02, 0.0, 0.03],
        com_offset_body=(0.4, -0.1, 0.05),
        coupling_mode="natural_contact_verified_grasp_ramped_payload_v2",
    )

    selected = _diagnostic_payload_coupling_component_view(
        full,
        component_mode=mode,
    )

    assert selected.mass_kg == pytest.approx(full.mass_kg)
    assert selected.gravity_mps2 == pytest.approx(full.gravity_mps2)
    assert selected.com_offset_body == pytest.approx(expected_offset)
    assert selected.inertia_body == pytest.approx(expected_inertia)
    assert _diagnostic_payload_coupling_component_flags(mode) == expected_flags


def test_diagnostic_payload_component_view_rejects_unknown_mode() -> None:
    coupling = PayloadCoupling(
        payload_id="box_01",
        contact_model="natural_contact_grasp_v1",
        mass_kg=1.0,
        inertia_body=[0.0] * 6,
        com_offset_body=(0.0, 0.0, 0.0),
    )

    with pytest.raises(SchemaValidationError, match="component mode"):
        _diagnostic_payload_coupling_component_view(
            coupling,
            component_mode="unknown",
        )


def test_natural_contact_payload_coupling_rotates_inertia_and_zero_disables() -> None:
    half_sqrt = math.sqrt(0.5)
    coupling = _natural_contact_payload_coupling(
        control_body_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        object_com_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, half_sqrt, half_sqrt),
        object_mass_kg=1.0,
        object_size_m=(0.3, 0.4, 0.15),
        load_transfer_scale=1.0,
        contact_model="natural_contact_grasp_v1",
    )

    assert coupling is not None
    object_ixx = (0.4**2 + 0.15**2) / 12.0
    object_iyy = (0.3**2 + 0.15**2) / 12.0
    assert coupling.inertia_body[0] == pytest.approx(object_iyy)
    assert coupling.inertia_body[3] == pytest.approx(object_ixx)
    assert (
        _natural_contact_payload_coupling(
            control_body_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            object_com_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            object_mass_kg=1.0,
            object_size_m=(0.3, 0.4, 0.15),
            load_transfer_scale=0.0,
            contact_model="natural_contact_grasp_v1",
        )
        is None
    )


@pytest.mark.parametrize("scale", [-0.1, 1.1, math.nan])
def test_natural_contact_payload_coupling_rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(SchemaValidationError, match="load-transfer scale"):
        _natural_contact_payload_coupling(
            control_body_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            object_com_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            object_mass_kg=1.0,
            object_size_m=(0.3, 0.4, 0.15),
            load_transfer_scale=scale,
            contact_model="natural_contact_grasp_v1",
        )


@pytest.mark.parametrize(
    ("phase", "elapsed_s", "expected_scale"),
    [
        (Order8NaturalContactPhase.CONTACT_ACQUISITION, 0.0, 1.0),
        (Order8NaturalContactPhase.LIFT, 0.0, 0.0),
        (Order8NaturalContactPhase.LIFT, 0.125, 0.5),
        (Order8NaturalContactPhase.LIFT, 0.25, 1.0),
        (Order8NaturalContactPhase.LIFT, 0.375, 1.0),
        (Order8NaturalContactPhase.LIFT, 0.50, 1.0),
        (Order8NaturalContactPhase.TRANSPORT, 0.05, 0.2),
        (Order8NaturalContactPhase.PLACE, 1.0, 1.0),
        (Order8NaturalContactPhase.RELEASE, 0.0, 1.0),
    ],
)
def test_contact_motion_entry_speed_ramps_maintained_contact_phases(
    phase: Order8NaturalContactPhase,
    elapsed_s: float,
    expected_scale: float,
) -> None:
    assert _contact_motion_entry_speed_scale(
        phase,
        phase_elapsed_s=elapsed_s,
        transition_duration_s=0.25,
    ) == pytest.approx(expected_scale)


@pytest.mark.parametrize(
    ("elapsed_s", "duration_s"),
    [(-0.001, 0.25), (math.nan, 0.25), (0.0, 0.0), (0.0, math.inf)],
)
def test_contact_motion_entry_speed_rejects_invalid_timing(
    elapsed_s: float,
    duration_s: float,
) -> None:
    with pytest.raises(SchemaValidationError, match="contact-motion"):
        _contact_motion_entry_speed_scale(
            Order8NaturalContactPhase.LIFT,
            phase_elapsed_s=elapsed_s,
            transition_duration_s=duration_s,
        )


@pytest.mark.parametrize(
    ("nominal_ready", "privileged_grasp", "expected"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_contact_required_motion_requires_nominal_and_safety_dwell(
    nominal_ready: bool,
    privileged_grasp: bool,
    expected: bool,
) -> None:
    assert (
        _contact_required_motion_safety_authorized(
            nominal_command_dwell_complete=nominal_ready,
            privileged_grasp_dwell_acquired=privileged_grasp,
        )
        is expected
    )


def test_contact_anchor_reference_uses_measured_orientation_only() -> None:
    desired = {
        0: (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9),
        1: (-1.0, -2.0, -3.0, 0.4, 0.5, 0.6, 0.7),
    }
    measured = {
        0: (9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0),
        1: (6.0, 5.0, 4.0, 0.0, 0.0, 1.0, 0.0),
    }

    references = _contact_anchor_references_with_measured_orientation(
        desired,
        measured,
    )

    assert references == {
        0: (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
        1: (-1.0, -2.0, -3.0, 0.0, 0.0, 1.0, 0.0),
    }


def test_contact_anchor_reference_requires_exact_pose_coverage() -> None:
    with pytest.raises(SchemaValidationError, match="identical non-empty coverage"):
        _contact_anchor_references_with_measured_orientation(
            {0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
            {1: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
        )


def test_dock_armature_override_is_diagnostic_only_and_keeps_config_separate() -> None:
    configured = {"armature_kg_m2": 0.002}

    assert _dock_joint_armature_setting(
        simulation_drive=configured,
        diagnostic_override_kg_m2=None,
        diagnostic_only=False,
    ) == (
        0.002,
        None,
        "joint_actuator_simulation_drive_armature_kg_m2_v1",
    )
    assert _dock_joint_armature_setting(
        simulation_drive=configured,
        diagnostic_override_kg_m2=0.01,
        diagnostic_only=True,
    ) == (
        0.01,
        0.01,
        "acceptance_ineligible_diagnostic_armature_override_v1",
    )
    assert configured == {"armature_kg_m2": 0.002}

    with pytest.raises(SchemaValidationError, match="diagnostic-only"):
        _dock_joint_armature_setting(
            simulation_drive=configured,
            diagnostic_override_kg_m2=0.01,
            diagnostic_only=False,
        )


def test_dock_armature_setting_rejects_invalid_values() -> None:
    with pytest.raises(SchemaValidationError, match="non-negative"):
        _dock_joint_armature_setting(
            simulation_drive={"armature_kg_m2": -0.1},
            diagnostic_override_kg_m2=None,
            diagnostic_only=False,
        )
    with pytest.raises(SchemaValidationError, match="positive"):
        _dock_joint_armature_setting(
            simulation_drive={},
            diagnostic_override_kg_m2=0.0,
            diagnostic_only=True,
        )


def test_centroidal_shape_uses_measured_state_not_unreached_actuator_target() -> None:
    measured = {
        "module_0:dock_a": 0.10,
        "module_1:dock_b": -0.20,
    }
    actuator = {
        "module_0:dock_a": 0.13,
        "module_1:dock_b": -0.24,
    }

    result = _centroidal_measured_joint_reference(
        expected_joint_ids=("module_0:dock_a", "module_1:dock_b"),
        actuator_position_targets=actuator,
        measured_joint_positions=measured,
    )

    assert result == measured
    assert result is not measured
    assert result != actuator


def test_centroidal_shape_reference_requires_exact_finite_joint_coverage() -> None:
    with pytest.raises(SchemaValidationError, match="cover exactly"):
        _centroidal_measured_joint_reference(
            expected_joint_ids=("module_0:dock",),
            actuator_position_targets={"module_0:dock": 0.1},
            measured_joint_positions={"module_1:dock": 0.1},
        )
    with pytest.raises(SchemaValidationError, match="finite"):
        _centroidal_measured_joint_reference(
            expected_joint_ids=("module_0:dock",),
            actuator_position_targets={"module_0:dock": math.nan},
            measured_joint_positions={"module_0:dock": 0.1},
        )


@pytest.mark.parametrize(
    ("phase", "elapsed_s", "expected"),
    [
        (Order8NaturalContactPhase.CONTACT_ACQUISITION, 10.0, 0.25),
        (Order8NaturalContactPhase.LIFT, 0.0, 1.0),
        (Order8NaturalContactPhase.TRANSPORT, 0.0, 1.0),
        (Order8NaturalContactPhase.PLACE, 0.0, 1.0),
        (Order8NaturalContactPhase.RELEASE, 40.0, 0.0),
    ],
)
def test_contact_force_ramps_then_holds_until_planned_release(
    phase: Order8NaturalContactPhase,
    elapsed_s: float,
    expected: float,
) -> None:
    assert _contact_force_scale_for_phase(
        phase=phase,
        ramp_elapsed_s=elapsed_s,
        ramp_duration_s=40.0,
    ) == pytest.approx(expected)


def test_contact_force_hold_motion_gate_uses_only_selected_clearance_rate() -> None:
    assert _contact_force_hold_settled(
        {0: 0.0010, 1: 0.0015, 99: 4.0},
        selected_anchor_ids=(0, 1),
        speed_threshold_mps=0.0015,
    )


def test_prelift_relative_speed_threshold_reserves_slip_margin() -> None:
    assert _prelift_relative_speed_threshold_mps(
        maintained_contact_slip_limit_mps=0.02
    ) == pytest.approx(0.01)
    with pytest.raises(SchemaValidationError, match="slip limit"):
        _prelift_relative_speed_threshold_mps(
            maintained_contact_slip_limit_mps=0.0
        )


def test_dock_torque_bias_uses_continuous_limit_below_hard_peak() -> None:
    joint = SimpleNamespace(
        limit_lower=-1.0,
        limit_upper=1.0,
        velocity_limit=38.0,
        effort_limit=4.1,
    )
    dock_spec = {
        "continuous_torque_limit_nm": 1.3,
        "peak_torque_nm": 4.1,
        "peak_current_a": 7.3,
        "simulation_drive": {"safe_velocity_limit_rad_s": 3.0},
    }

    limit = _dock_limit(joint, dock_spec)

    assert limit.max_torque_nm == pytest.approx(1.3)
    assert limit.max_velocity_radps == pytest.approx(3.0)
    contact_limit = _dock_limit(
        joint,
        dock_spec,
        velocity_limit_override_radps=0.1,
    )
    assert contact_limit.max_velocity_radps == pytest.approx(0.1)
    with pytest.raises(SchemaValidationError, match="finite and positive"):
        _dock_limit(
            joint,
            dock_spec,
            velocity_limit_override_radps=0.0,
        )
    assert not _contact_force_hold_settled(
        {0: 0.0010, 1: 0.0015001},
        selected_anchor_ids=(0, 1),
        speed_threshold_mps=0.0015,
    )


def test_diagnostic_peak_torque_window_returns_smoothly_to_continuous() -> None:
    kwargs = {
        "continuous_torque_nm": 1.3,
        "peak_torque_nm": 4.1,
        "peak_window_s": 1.5,
    }

    assert _torque_bias_limit_with_peak_window(
        **kwargs, elapsed_since_qclose_s=None
    ) == pytest.approx(1.3)
    assert _torque_bias_limit_with_peak_window(
        **kwargs, elapsed_since_qclose_s=0.0
    ) == pytest.approx(4.1)
    assert _torque_bias_limit_with_peak_window(
        **kwargs, elapsed_since_qclose_s=1.25
    ) == pytest.approx(4.1)
    assert _torque_bias_limit_with_peak_window(
        **kwargs, elapsed_since_qclose_s=1.375
    ) == pytest.approx(2.7)
    assert _torque_bias_limit_with_peak_window(
        **kwargs, elapsed_since_qclose_s=1.5
    ) == pytest.approx(1.3)
    assert _torque_bias_limit_with_peak_window(
        **kwargs, elapsed_since_qclose_s=1.5 - 1.0e-14
    ) == 1.3
    assert _torque_bias_limit_with_peak_window(
        **kwargs, elapsed_since_qclose_s=1.5 - 1.0e-13
    ) == 1.3
    assert _torque_bias_limit_with_peak_window(
        continuous_torque_nm=1.3,
        peak_torque_nm=4.1,
        elapsed_since_qclose_s=0.0,
        peak_window_s=None,
    ) == pytest.approx(1.3)


def test_peak_torque_window_synchronizes_only_dock_effort_bias_channels() -> None:
    mapping = ActuatorMapping(
        graph_id="graph",
        module_ids=[0],
        channels=[
            ActuatorChannel(
                actuator_id="module_0:dock",
                module_id=0,
                local_id="dock",
                actuator_type="dock_joint_position",
                isaac_target_name="module_0/dock",
                effort=4.1,
                supported_command_types=[
                    "joint_position",
                    "joint_velocity",
                    "joint_effort_bias",
                ],
                metadata={
                    "continuous_torque_limit_nm": 1.3,
                    "peak_torque_limit_nm": 4.1,
                },
            ),
            ActuatorChannel(
                actuator_id="module_0:rotor",
                module_id=0,
                local_id="rotor",
                actuator_type="rotor_thrust",
                isaac_target_name="module_0/rotor",
                lower=0.0,
                upper=20.0,
            ),
        ],
        command_key_aliases={
            "module_0:dock": "module_0:dock",
            "module_0:rotor": "module_0:rotor",
        },
    )

    active = _actuator_mapping_with_torque_bias_limit(
        mapping,
        active_limit_nm=2.9,
    )

    assert active is not mapping
    assert active.channels[0].metadata["continuous_torque_limit_nm"] == pytest.approx(
        2.9
    )
    assert mapping.channels[0].metadata["continuous_torque_limit_nm"] == pytest.approx(
        1.3
    )
    assert active.channels[1] == mapping.channels[1]
    with pytest.raises(SchemaValidationError, match="outside the recorded"):
        _actuator_mapping_with_torque_bias_limit(mapping, active_limit_nm=4.2)


def test_signed_normal_velocity_filter_cancels_contact_micro_oscillation() -> None:
    value = 0.0
    for sample in (0.004, -0.004) * 100:
        value = _first_order_low_pass(
            value,
            sample,
            dt_s=0.005,
            time_constant_s=0.1,
        )

    assert abs(value) < 0.00011
    with pytest.raises(ValueError, match="positive"):
        _first_order_low_pass(0.0, 1.0, dt_s=0.0, time_constant_s=0.1)


def test_qclose_checkpoint_state_json_roundtrips_exact_dynamic_state() -> None:
    payload = {
        "schema_version": "order8_qclose_checkpoint_state_v1",
        "module_root_poses": {"0": [0, 0, 1, 0, 0, 0, 1]},
        "module_root_velocities": {"0": [1, 2, 3, 4, 5, 6]},
        "joint_positions_rad": {"module_0:dock": 0.1},
        "joint_velocities_radps": {"module_0:dock": -0.2},
        "object_pose": [1, 0, 0.5, 0, 0, 0, 1],
        "object_twist": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "anchor_hold_poses_base": {"7": [0.3, 0, 0, 0, 0, 0, 1]},
    }

    state = _parse_qclose_checkpoint_state(json.dumps(payload))

    assert state is not None
    assert state.module_root_velocities[0] == pytest.approx((1, 2, 3, 4, 5, 6))
    assert state.joint_velocities_radps == {"module_0:dock": -0.2}
    assert _qclose_checkpoint_state_to_dict(state) == payload


def test_qclose_checkpoint_state_rejects_incomplete_or_wrong_version() -> None:
    with pytest.raises(RuntimeError, match="version mismatch"):
        _parse_qclose_checkpoint_state(json.dumps({"schema_version": "wrong"}))
    with pytest.raises(RuntimeError, match="module_root_velocities"):
        _parse_qclose_checkpoint_state(
            json.dumps(
                {
                    "schema_version": "order8_qclose_checkpoint_state_v1",
                    "module_root_poses": {"0": [0, 0, 1, 0, 0, 0, 1]},
                }
            )
        )


def test_order8_seed_is_applied_and_reported_deterministically() -> None:
    import random

    import numpy

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

        def __init__(self) -> None:
            self.seeds: list[int] = []

        def manual_seed(self, seed: int) -> None:
            self.seeds.append(seed)

    torch = FakeTorch()
    first = _apply_order8_seed(17, torch=torch)
    first_samples = (random.random(), float(numpy.random.random()))
    second = _apply_order8_seed(17, torch=torch)
    second_samples = (random.random(), float(numpy.random.random()))

    assert (
        first
        == second
        == {
            "seed": 17,
            "python_random": True,
            "torch": True,
            "torch_cuda": False,
            "numpy": True,
        }
    )
    assert torch.seeds == [17, 17]
    assert first_samples == second_samples


def test_whole_structure_runtime_observation_covers_one_connected_graph() -> None:
    graph = SimpleNamespace(
        modules=[SimpleNamespace(module_id=2), SimpleNamespace(module_id=0)]
    )
    states = {
        module_id: ModuleRuntimeState(
            module_id=module_id,
            pose_world=(float(module_id), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            twist_world=[0.0] * 6,
            joint_positions={"dock": 0.1 * module_id},
            joint_velocities={"dock": 0.0},
        )
        for module_id in (2, 0)
    }
    status = ControllerStatus(status="ok", qp_feasible=True)

    observation = _whole_structure_runtime_observation(
        time_s=1.25,
        morphology_graph=graph,
        module_states_by_id=states,
        controller_status=status,
        phase_label="contact_acquisition",
    )

    assert observation.morphology_graph is graph
    assert [state.module_id for state in observation.module_states] == [0, 2]
    assert observation.controller_status is status
    assert observation.task_progress.phase_label == "contact_acquisition"

    with pytest.raises(SchemaValidationError, match="cover exactly"):
        _whole_structure_runtime_observation(
            time_s=1.25,
            morphology_graph=graph,
            module_states_by_id={0: states[0]},
            controller_status=status,
            phase_label="contact_acquisition",
        )
    with pytest.raises(SchemaValidationError, match="state ids"):
        _whole_structure_runtime_observation(
            time_s=1.25,
            morphology_graph=graph,
            module_states_by_id={0: states[2], 2: states[0]},
            controller_status=status,
            phase_label="contact_acquisition",
        )


def test_dock_actuator_telemetry_separates_position_drive_and_torque_bias() -> None:
    telemetry = _dock_joint_actuator_telemetry_entry(
        requested_position_target_rad=0.11,
        requested_velocity_target_radps=0.0,
        requested_unclipped_torque_bias_nm=3.0,
        requested_limited_torque_bias_nm=1.3,
        measured_position_rad=0.10,
        measured_velocity_radps=0.20,
        isaac_position_target_rad=0.11,
        isaac_velocity_target_radps=0.0,
        isaac_effort_target_nm=1.3,
        isaac_computed_torque_nm=1.3,
        isaac_applied_torque_nm=1.3,
        stiffness_nm_per_rad=200.0,
        damping_nms_per_rad=2.0,
        effort_limit_sim_nm=4.1,
        peak_torque_nm=4.1,
        peak_current_a=7.3,
    )

    assert telemetry["position_error_rad"] == pytest.approx(0.01)
    assert telemetry["velocity_error_radps"] == pytest.approx(-0.20)
    assert telemetry["estimated_position_drive_torque_nm"] == pytest.approx(2.0)
    assert telemetry["estimated_damping_drive_torque_nm"] == pytest.approx(-0.4)
    assert telemetry["estimated_total_drive_torque_nm"] == pytest.approx(2.9)
    assert telemetry["estimated_current_a"] == pytest.approx(1.3 / 4.1 * 7.3)
    assert telemetry["torque_bias_limited"] is True


def test_latched_qclose_holds_absolute_position_and_zero_velocity() -> None:
    @dataclass(frozen=True)
    class FakeJointResult:
        policy_command: PolicyCommand

    result = FakeJointResult(
        policy_command=PolicyCommand(
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets={"module_0:a": 0.55, "module_0:b": -0.45},
            joint_velocity_targets={"module_0:a": 0.2, "module_0:b": -0.2},
            joint_torque_bias={"module_0:a": 1.0, "module_0:b": -1.0},
        )
    )
    joint_vector = SimpleNamespace(joint_ids=("module_0:a", "module_0:b"))

    held = _hold_latched_joint_positions(
        result,
        joint_vector,
        position_reference_rad={"module_0:a": 0.5, "module_0:b": -0.5},
    )

    assert held.policy_command.joint_position_targets == {
        "module_0:a": 0.5,
        "module_0:b": -0.5,
    }
    assert held.policy_command.joint_velocity_targets == {
        "module_0:a": 0.0,
        "module_0:b": 0.0,
    }
    assert held.policy_command.joint_torque_bias == {
        "module_0:a": 1.0,
        "module_0:b": -1.0,
    }
    assert result.policy_command.joint_position_targets["module_0:a"] == 0.55

    with pytest.raises(SchemaValidationError, match="cover exactly"):
        _hold_latched_joint_positions(
            result,
            joint_vector,
            position_reference_rad={"module_0:a": 0.5},
        )


def test_loaded_state_rebase_settle_dwell_requires_continuous_relative_settle() -> None:
    dwell = _advance_loaded_state_rebase_settle_dwell(
        0.0,
        relative_speed_mps_by_anchor={0: 0.004, 1: 0.009},
        selected_anchor_ids=(0, 1),
        speed_threshold_mps=0.010,
        dt_s=0.02,
    )
    assert dwell == pytest.approx(0.02)

    dwell = _advance_loaded_state_rebase_settle_dwell(
        dwell,
        relative_speed_mps_by_anchor={0: 0.011, 1: 0.003},
        selected_anchor_ids=(0, 1),
        speed_threshold_mps=0.010,
        dt_s=0.02,
    )
    assert dwell == pytest.approx(0.0)

    with pytest.raises(SchemaValidationError, match="finite and positive"):
        _advance_loaded_state_rebase_settle_dwell(
            0.0,
            relative_speed_mps_by_anchor={0: 0.0, 1: 0.0},
            selected_anchor_ids=(0, 1),
            speed_threshold_mps=0.010,
            dt_s=0.0,
        )


def test_loaded_state_rebase_suppresses_only_transient_acceleration_bias() -> None:
    assert _loaded_state_rebase_acceleration_bias_scale(
        0.75,
        rebase_settle_active=True,
    ) == pytest.approx(0.0)
    assert _loaded_state_rebase_acceleration_bias_scale(
        0.75,
        rebase_settle_active=False,
    ) == pytest.approx(0.75)

    with pytest.raises(SchemaValidationError, match=r"in \[0, 1\]"):
        _loaded_state_rebase_acceleration_bias_scale(
            1.01,
            rebase_settle_active=True,
        )


def test_contact_joint_drive_impedance_schedule_resolves_all_dock_joints() -> None:
    class FakeRobot:
        def __init__(self, joint_names: list[str]) -> None:
            self.joint_names = joint_names
            self.stiffness_calls: list[tuple[float, list[int]]] = []
            self.damping_calls: list[tuple[float, list[int]]] = []

        def write_joint_stiffness_to_sim_index(
            self,
            *,
            stiffness: float,
            joint_ids: list[int],
        ) -> None:
            self.stiffness_calls.append((stiffness, joint_ids))

        def write_joint_damping_to_sim_index(
            self,
            *,
            damping: float,
            joint_ids: list[int],
        ) -> None:
            self.damping_calls.append((damping, joint_ids))

    robots = {
        0: FakeRobot(["gimbal1", "pitch_dock_mech_joint1"]),
        1: FakeRobot(["yaw_dock_mech_joint2", "gimbal2"]),
    }
    expected = (
        "module_0:pitch_dock_mech_joint1",
        "module_1:yaw_dock_mech_joint2",
    )

    stiffness_targets, damping_targets = _schedule_contact_joint_drive_impedance(
        robots,
        expected,
        stiffness_nm_per_rad=50.0,
        damping_nms_per_rad=8.0,
        maximum_stiffness_nm_per_rad=200.0,
        maximum_damping_nms_per_rad=50.0,
    )

    assert stiffness_targets == {joint_id: 50.0 for joint_id in expected}
    assert damping_targets == {joint_id: 8.0 for joint_id in expected}
    assert robots[0].stiffness_calls == [(50.0, [1])]
    assert robots[1].stiffness_calls == [(50.0, [0])]
    assert robots[0].damping_calls == [(8.0, [1])]
    assert robots[1].damping_calls == [(8.0, [0])]

    with pytest.raises(SchemaValidationError, match="stiffness exceeds"):
        _schedule_contact_joint_drive_impedance(
            robots,
            expected,
            stiffness_nm_per_rad=200.1,
            damping_nms_per_rad=8.0,
            maximum_stiffness_nm_per_rad=200.0,
            maximum_damping_nms_per_rad=50.0,
        )
    with pytest.raises(SchemaValidationError, match="resolve"):
        _schedule_contact_joint_drive_impedance(
            robots,
            ("module_0:missing",),
            stiffness_nm_per_rad=50.0,
            damping_nms_per_rad=8.0,
            maximum_stiffness_nm_per_rad=200.0,
            maximum_damping_nms_per_rad=50.0,
        )


def test_contact_joint_drive_damping_schedule_resolves_all_dock_joints() -> None:
    class FakeRobot:
        def __init__(self, joint_names: list[str]) -> None:
            self.joint_names = joint_names
            self.calls: list[tuple[float, list[int]]] = []

        def write_joint_damping_to_sim_index(
            self,
            *,
            damping: float,
            joint_ids: list[int],
        ) -> None:
            self.calls.append((damping, joint_ids))

    robots = {
        0: FakeRobot(["gimbal1", "pitch_dock_mech_joint1"]),
        1: FakeRobot(["yaw_dock_mech_joint2", "gimbal2"]),
    }
    expected = (
        "module_0:pitch_dock_mech_joint1",
        "module_1:yaw_dock_mech_joint2",
    )

    targets = _schedule_contact_joint_drive_damping(
        robots,
        expected,
        nominal_damping_nms_per_rad=2.0,
        damping_multiplier=2.5,
        maximum_damping_nms_per_rad=5.0,
    )

    assert targets == {joint_id: 5.0 for joint_id in expected}
    assert robots[0].calls == [(5.0, [1])]
    assert robots[1].calls == [(5.0, [0])]

    with pytest.raises(SchemaValidationError, match="at least one"):
        _schedule_contact_joint_drive_damping(
            robots,
            expected,
            nominal_damping_nms_per_rad=2.0,
            damping_multiplier=0.99,
            maximum_damping_nms_per_rad=5.0,
        )
    with pytest.raises(SchemaValidationError, match="simulation limit"):
        _schedule_contact_joint_drive_damping(
            robots,
            expected,
            nominal_damping_nms_per_rad=2.0,
            damping_multiplier=2.51,
            maximum_damping_nms_per_rad=5.0,
        )
    with pytest.raises(SchemaValidationError, match="resolve"):
        _schedule_contact_joint_drive_damping(
            robots,
            ("module_0:missing",),
            nominal_damping_nms_per_rad=2.0,
            damping_multiplier=2.0,
            maximum_damping_nms_per_rad=5.0,
        )


def test_module_frame_state_uses_named_link_instead_of_articulation_root() -> None:
    class FakeRow:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self.values)

    class FakeTensor:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, index):
            batch, body = index
            assert batch == 0
            return FakeRow(self.rows[body])

    robot = SimpleNamespace(
        body_names=["main_body", "module_0__fc"],
        data=SimpleNamespace(
            body_pos_w=FakeTensor([(9.0, 9.0, 9.0), (1.0, 2.0, 3.0)]),
            body_quat_w=FakeTensor([(0.0, 0.0, 0.0, 1.0), (0.1, 0.2, 0.3, 0.9)]),
            body_lin_vel_w=FakeTensor([(8.0, 8.0, 8.0), (0.4, 0.5, 0.6)]),
            body_ang_vel_w=FakeTensor([(7.0, 7.0, 7.0), (0.7, 0.8, 0.9)]),
        ),
    )

    pose, twist = _module_frame_pose_twist(
        robot,
        module_frame_link_id="fc",
    )

    assert pose == pytest.approx((1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9))
    assert twist == pytest.approx((0.4, 0.5, 0.6, 0.7, 0.8, 0.9))


def _selection(
    anchor_id: int, normal: tuple[float, float, float]
) -> NaturalContactAnchorSelection:
    return NaturalContactAnchorSelection(
        anchor_id=anchor_id,
        slot_id=anchor_id,
        candidate_id=anchor_id,
        dock_link_id=f"module_{anchor_id}:dock",
        inward_normal_world=normal,
    )


def test_anchor_linearization_preserves_full_whole_structure_jacobian() -> None:
    joint_ids = (
        "module_0:pitch_dock_joint",
        "module_0:yaw_dock_joint",
        "module_1:pitch_dock_joint",
    )
    first_jacobian = (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
        (0.7, 0.8, 0.9),
        (1.0, 1.1, 1.2),
        (1.3, 1.4, 1.5),
        (1.6, 1.7, 1.8),
    )
    second_jacobian = tuple(tuple(-value for value in row) for row in first_jacobian)
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    kinematics = SimpleNamespace(
        ordered_global_dock_joint_ids=joint_ids,
        anchor_poses_world={4: identity, 9: identity},
        anchor_jacobians={4: first_jacobian, 9: second_jacobian},
    )

    tasks = _anchor_task_linearizations(
        kinematics,
        [_selection(9, (-2.0, 0.0, 0.0)), _selection(4, (0.0, 3.0, 0.0))],
        desired_anchor_poses={
            4: (0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0),
            9: (-0.4, 0.5, -0.6, 0.0, 0.0, 0.0, 1.0),
        },
        wrench_targets={
            4: (0.0, 11.0, 0.0, 0.0, 0.0, 0.0),
            9: (-11.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
    )

    assert [task.anchor_id for task in tasks] == [4, 9]
    assert tasks[0].jacobian == first_jacobian
    assert tasks[1].jacobian == second_jacobian
    assert all(len(row) == len(joint_ids) for task in tasks for row in task.jacobian)
    assert tasks[0].task_error == pytest.approx((0.1, -0.2, 0.3, 0.0, 0.0, 0.0))
    assert tasks[0].wrench_bias == pytest.approx((0.0, 11.0, 0.0, 0.0, 0.0, 0.0))
    assert tasks[1].wrench_bias == pytest.approx((-11.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_anchor_linearization_shifts_mesh_point_force_to_connect_frame() -> None:
    current = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    kinematics = SimpleNamespace(
        ordered_global_dock_joint_ids=("module_0:pitch_dock_joint",),
        anchor_poses_world={4: current},
        anchor_jacobians={4: tuple((1.0,) for _ in range(6))},
    )

    task = _anchor_task_linearizations(
        kinematics,
        [_selection(4, (0.0, 1.0, 0.0))],
        desired_anchor_poses={4: current},
        wrench_targets={4: (0.0, 2.0, 0.0, 0.1, 0.2, 0.3)},
        wrench_application_points_world={4: (1.0, 2.0, 4.0)},
    )[0]

    # r=(0,0,1), F=(0,2,0), so r x F=(-2,0,0).  The shift
    # preserves the commanded contact-point force while expressing its moment
    # at the connect-frame origin used by the whole-structure Jacobian.
    assert task.wrench_bias == pytest.approx((0.0, 2.0, 0.0, -1.9, 0.2, 0.3))


def test_anchor_linearization_uses_same_surface_point_for_error_and_jacobian() -> (
    None
):
    current = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    desired = (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    kinematics = SimpleNamespace(
        ordered_global_dock_joint_ids=("module_0:pitch_dock_joint",),
        anchor_poses_world={4: current},
        anchor_jacobians={
            4: (
                (0.0,),
                (0.0,),
                (0.0,),
                (0.0,),
                (0.0,),
                (1.0,),
            )
        },
    )

    task = _anchor_task_linearizations(
        kinematics,
        [_selection(4, (1.0, 0.0, 0.0))],
        desired_anchor_poses={4: desired},
        wrench_targets={4: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        current_anchor_poses_world={4: current},
        task_application_points_world={4: (0.0, 1.0, 0.0)},
    )[0]

    assert task.task_error == pytest.approx((0.1, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert task.jacobian[0] == pytest.approx((-1.0,))
    assert task.jacobian[1] == pytest.approx((0.0,))


def test_anchor_linearization_accepts_direct_surface_point_target() -> None:
    current = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    kinematics = SimpleNamespace(
        ordered_global_dock_joint_ids=("module_0:pitch_dock_joint",),
        anchor_poses_world={4: current},
        anchor_jacobians={4: tuple((0.0,) for _ in range(6))},
    )

    task = _anchor_task_linearizations(
        kinematics,
        [_selection(4, (1.0, 0.0, 0.0))],
        desired_anchor_poses={4: current},
        wrench_targets={4: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        current_anchor_poses_world={4: current},
        task_application_points_world={4: (0.0, 1.0, 0.0)},
        desired_task_application_points_world={4: (0.0, 0.9, 0.0)},
    )[0]

    assert task.task_error == pytest.approx((0.0, -0.1, 0.0, 0.0, 0.0, 0.0))


def test_anchor_linearization_requires_one_finite_application_point_per_anchor() -> (
    None
):
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    kinematics = SimpleNamespace(
        ordered_global_dock_joint_ids=("module_0:pitch_dock_joint",),
        anchor_poses_world={0: identity, 1: identity},
        anchor_jacobians={
            0: tuple((0.0,) for _ in range(6)),
            1: tuple((0.0,) for _ in range(6)),
        },
    )
    selections = [
        _selection(0, (1.0, 0.0, 0.0)),
        _selection(1, (-1.0, 0.0, 0.0)),
    ]
    kwargs = {
        "desired_anchor_poses": {0: identity, 1: identity},
        "wrench_targets": {
            0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            1: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
    }

    with pytest.raises(SchemaValidationError, match="cover exactly"):
        _anchor_task_linearizations(
            kinematics,
            selections,
            **kwargs,
            wrench_application_points_world={0: (0.0, 0.0, 0.0)},
        )
    with pytest.raises(SchemaValidationError, match="three finite"):
        _anchor_task_linearizations(
            kinematics,
            selections,
            **kwargs,
            wrench_application_points_world={
                0: (0.0, 0.0, 0.0),
                1: (math.nan, 0.0, 0.0),
            },
        )


def test_anchor_linearization_scales_pose_task_from_planner_priority() -> None:
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    kinematics = SimpleNamespace(
        ordered_global_dock_joint_ids=("module_0:pitch_dock_joint",),
        anchor_poses_world={0: identity, 1: identity},
        anchor_jacobians={
            0: tuple((1.0,) for _ in range(6)),
            1: tuple((1.0,) for _ in range(6)),
        },
    )

    tasks = _anchor_task_linearizations(
        kinematics,
        [
            _selection(0, (1.0, 0.0, 0.0)),
            _selection(1, (-1.0, 0.0, 0.0)),
        ],
        desired_anchor_poses={0: identity, 1: identity},
        wrench_targets={
            0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            1: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
        task_priorities={0: 0.05, 1: 1.0},
        orientation_task_weight=1.0,
    )

    assert tasks[0].task_weights == pytest.approx((0.05, 0.05, 0.05, 0.05, 0.05, 0.05))
    assert tasks[1].task_weights == pytest.approx((1.0, 1.0, 1.0, 1.0, 1.0, 1.0))


def test_anchor_linearization_uses_world_frame_rotation_log_error() -> None:
    half = 0.5 * math.pi / 2.0
    kinematics = SimpleNamespace(
        ordered_global_dock_joint_ids=("module_0:pitch_dock_joint",),
        anchor_poses_world={0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
        anchor_jacobians={0: tuple((0.0,) for _ in range(6))},
    )

    task = _anchor_task_linearizations(
        kinematics,
        [_selection(0, (1.0, 0.0, 0.0))],
        desired_anchor_poses={
            0: (0.0, 0.0, 0.0, 0.0, 0.0, math.sin(half), math.cos(half))
        },
        wrench_targets={0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
    )[0]

    assert task.task_error[3:] == pytest.approx((0.0, 0.0, math.pi / 2.0))


def test_planner_active_knot_is_the_only_nominal_anchor_and_wrench_source() -> None:
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    desired = (0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0)
    kinematics = SimpleNamespace(
        ordered_global_dock_joint_ids=("module_0:pitch_dock_joint",),
        anchor_poses_world={3: identity},
        anchor_jacobians={3: tuple((float(index),) for index in range(6))},
    )
    trajectory = SimpleNamespace(
        knots=[
            SimpleNamespace(
                centroidal_target=SimpleNamespace(
                    com_pos_world=(1.0, 2.0, 3.0),
                    com_vel_world=(0.2, -0.1, 0.0),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                ),
                posture_target=SimpleNamespace(free_anchor_pose_targets={3: desired}),
                contact_assignments=[
                    SimpleNamespace(
                        anchor_id=3,
                        wrench_target=[1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                    )
                ],
            )
        ]
    )

    task = _anchor_tasks_from_planner_trajectory(
        kinematics,
        [_selection(3, (1.0, 0.0, 0.0))],
        trajectory,
    )[0]

    assert task.task_error == pytest.approx((0.2, -0.1, 0.3, 0.0, 0.0, 0.0))
    assert task.wrench_bias == pytest.approx((1.0, 2.0, 3.0, 0.1, 0.2, 0.3))
    assert _base_target_from_planner_trajectory(trajectory) == (
        1.0,
        2.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert _base_twist_from_planner_trajectory(trajectory) == (
        0.2,
        -0.1,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_global_dock_position_map_requires_one_position_per_unique_id() -> None:
    vector = SimpleNamespace(
        joint_ids=("module_0:pitch_dock_joint", "module_1:yaw_dock_joint"),
        positions_rad=(0.25, -0.5),
    )
    assert _global_dock_position_map(vector) == {
        "module_0:pitch_dock_joint": 0.25,
        "module_1:yaw_dock_joint": -0.5,
    }

    with pytest.raises(SchemaValidationError, match="same length"):
        _global_dock_position_map(
            SimpleNamespace(joint_ids=("joint",), positions_rad=())
        )
    with pytest.raises(SchemaValidationError, match="unique"):
        _global_dock_position_map(
            SimpleNamespace(joint_ids=("joint", "joint"), positions_rad=(0.0, 0.0))
        )


def test_advance_pose_toward_rate_limits_centroidal_translation() -> None:
    start = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    target = (3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    first = _advance_pose_toward(
        start,
        target,
        max_translation_step_m=0.5,
    )
    assert first[:3] == pytest.approx((0.3, 0.4, 0.0))
    assert first[3:] == target[3:]
    assert _advance_pose_toward(
        start,
        (0.1, 0.0, 0.0, *target[3:]),
        max_translation_step_m=0.5,
    ) == pytest.approx((0.1, 0.0, 0.0, *target[3:]))

    with pytest.raises(SchemaValidationError, match="finite and positive"):
        _advance_pose_toward(start, target, max_translation_step_m=0.0)


def test_alternating_reacquire_holds_fixed_in_band_snapshot() -> None:
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    stale_individual_latch = (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    terminal = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    # The first side retains its individual-arrest pose until the other side
    # has also arrested.
    assert (
        _alternating_reacquire_anchor_target(
            previous_command=identity,
            terminal_target=terminal,
            individual_latched_pose=stale_individual_latch,
            reacquired_hold_pose=None,
            all_individual_latches_acquired=False,
            max_translation_step_m=0.01,
        )
        == stale_individual_latch
    )

    # Once both sides have arrested, retain the one-time in-band snapshot so
    # coupled motion on the other side creates a restorative pose error.
    assert (
        _alternating_reacquire_anchor_target(
            previous_command=identity,
            terminal_target=terminal,
            individual_latched_pose=stale_individual_latch,
            reacquired_hold_pose=terminal,
            all_individual_latches_acquired=True,
            max_translation_step_m=0.01,
        )
        == terminal
    )

    # Only an out-of-band side continues toward the object-fixed terminal.
    assert _alternating_reacquire_anchor_target(
        previous_command=identity,
        terminal_target=terminal,
        individual_latched_pose=stale_individual_latch,
        reacquired_hold_pose=None,
        all_individual_latches_acquired=True,
        max_translation_step_m=0.01,
    ) == pytest.approx((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))


def test_sequential_reacquire_activates_one_task_without_masking_joint_columns() -> (
    None
):
    tasks = [
        SimpleNamespace(anchor_id=0, jacobian=((1.0, 2.0, 3.0),)),
        SimpleNamespace(anchor_id=1, jacobian=((4.0, 5.0, 6.0),)),
    ]

    assert (
        _sequential_reacquire_anchor_tasks(
            tasks,
            pursued_anchor_id=None,
        )
        == tasks
    )
    selected = _sequential_reacquire_anchor_tasks(
        tasks,
        pursued_anchor_id=1,
    )
    assert [task.anchor_id for task in selected] == [1]
    assert selected[0].jacobian[0] == (4.0, 5.0, 6.0)

    with pytest.raises(SchemaValidationError, match="exactly one task"):
        _sequential_reacquire_anchor_tasks(
            tasks,
            pursued_anchor_id=7,
        )


def test_sequential_latched_transfer_activates_only_world_hold_task() -> None:
    tasks = [
        SimpleNamespace(anchor_id=0, jacobian=((1.0, 2.0),)),
        SimpleNamespace(anchor_id=1, jacobian=((3.0, 4.0),)),
    ]

    selected = _sequential_latched_anchor_hold_tasks(
        tasks,
        latched_anchor_ids={1},
    )
    assert [task.anchor_id for task in selected] == [1]
    assert selected[0].jacobian[0] == (3.0, 4.0)

    with pytest.raises(SchemaValidationError, match="exactly one held anchor"):
        _sequential_latched_anchor_hold_tasks(
            tasks,
            latched_anchor_ids={0, 1},
        )


def test_sequential_centroidal_transfer_snapshots_clearance_with_bounded_margin() -> (
    None
):
    assert _sequential_centroidal_transfer_limit_m(
        observed_clearance_m=0.0052,
        clearance_margin_m=0.0015,
        maximum_transfer_m=0.030,
    ) == pytest.approx(0.0067)
    assert _sequential_centroidal_transfer_limit_m(
        observed_clearance_m=0.050,
        clearance_margin_m=0.0015,
        maximum_transfer_m=0.030,
    ) == pytest.approx(0.030)

    with pytest.raises(SchemaValidationError, match="finite and bounded"):
        _sequential_centroidal_transfer_limit_m(
            observed_clearance_m=-0.001,
            clearance_margin_m=0.0015,
            maximum_transfer_m=0.030,
        )


def test_contact_region_bounds_authored_mesh_sample_not_connect_frame() -> (
    None
):
    object_pose = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)
    nominal = (0.0, 0.20, 0.5, 0.0, 0.0, 0.0, 1.0)

    in_region = _contact_region_pose_target(
        current_anchor_pose_world=(0.04, 0.24, 0.47, 0.0, 0.0, 0.0, 1.0),
        current_surface_point_world=(0.08, 0.24, 0.43),
        nominal_anchor_pose_world=nominal,
        object_pose_world=object_pose,
        inward_normal_object=(0.0, -1.0, 0.0),
        tangential_tolerance_m=0.05,
    )
    # Both mesh-point tangent components are outside the +/-5 cm region.  The
    # anchor is translated only enough to put that physical point on the edge.
    assert in_region[:3] == pytest.approx((0.01, 0.20, 0.49))

    outside_region = _contact_region_pose_target(
        current_anchor_pose_world=(0.08, 0.24, 0.43, 0.0, 0.0, 0.0, 1.0),
        current_surface_point_world=(0.08, 0.24, 0.43),
        nominal_anchor_pose_world=nominal,
        object_pose_world=object_pose,
        inward_normal_object=(0.0, -1.0, 0.0),
        tangential_tolerance_m=0.05,
    )
    assert outside_region[:3] == pytest.approx((0.05, 0.20, 0.45))
    assert _contact_region_tangential_offsets_m(
        current_anchor_pose_world=(0.08, 0.20, 0.43, 0.0, 0.0, 0.0, 1.0),
        nominal_anchor_pose_world=(0.0, 0.20, 0.50, 0.0, 0.0, 0.0, 1.0),
        object_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        inward_normal_object=(0.0, -1.0, 0.0),
    ) == pytest.approx((0.08, -0.07))


def test_contact_precenter_retracts_overtravel_and_tracks_object_normal() -> None:
    # The normal contact point is already 4 mm inside the +Y face.  Precenter
    # removes that overtravel and retains 15 mm physical clearance.
    target = _contact_precenter_nominal_pose(
        (0.0, 0.196, 0.5, 0.0, 0.0, 0.0, 1.0),
        object_pose_world=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        inward_normal_object=(0.0, -1.0, 0.0),
        inward_overtravel_m=0.004,
        clearance_m=0.015,
    )
    assert target[:3] == pytest.approx((0.0, 0.215, 0.5))


def test_horizontal_mesh_pair_centering_balances_mean_without_vertical_shift() -> None:
    correction = _horizontal_mesh_pair_centering_correction_world(
        surface_point_world_by_anchor={
            0: (0.99, 0.2, 0.13),
            1: (0.93, -0.2, 0.11),
        },
        nominal_contact_pose_world_by_anchor={
            0: (1.0, 0.2, 0.075, 0.0, 0.0, 0.0, 1.0),
            1: (1.0, -0.2, 0.075, 0.0, 0.0, 0.0, 1.0),
        },
        approach_axis_world=(1.0, 0.0, 0.0),
        maximum_correction_m=0.05,
    )
    assert correction == pytest.approx((0.04, 0.0, 0.0))


def test_reference_pose_follows_free_object_translation_without_state_write() -> None:
    reference_object = (1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0)
    reference_target = (0.7, 2.1, 0.8, 0.0, 0.0, 0.0, 1.0)
    moved_object = (1.2, 1.9, 0.55, 0.0, 0.0, 0.0, 1.0)

    followed = _pose_following_object_motion(
        reference_object,
        moved_object,
        reference_target,
    )

    assert followed[:3] == pytest.approx((0.9, 2.0, 0.85))


def test_contact_anchor_priority_is_restored_for_coupled_reacquire() -> None:
    phase = Order8NaturalContactPhase.CONTACT_ACQUISITION
    assert _contact_anchor_pose_priority(
        phase=phase,
        contact_configuration_latched=False,
        anchor_individually_latched=True,
        all_individual_latches_acquired=False,
        anchor_reacquired=False,
        all_reacquired_holds_acquired=False,
    ) == pytest.approx(1.0)
    assert _contact_anchor_pose_priority(
        phase=phase,
        contact_configuration_latched=False,
        anchor_individually_latched=True,
        all_individual_latches_acquired=True,
        anchor_reacquired=True,
        all_reacquired_holds_acquired=False,
    ) == pytest.approx(0.5)
    assert _contact_anchor_pose_priority(
        phase=phase,
        contact_configuration_latched=False,
        anchor_individually_latched=True,
        all_individual_latches_acquired=True,
        anchor_reacquired=False,
        all_reacquired_holds_acquired=False,
    ) == pytest.approx(1.0)
    assert _contact_anchor_pose_priority(
        phase=phase,
        contact_configuration_latched=False,
        anchor_individually_latched=True,
        all_individual_latches_acquired=True,
        anchor_reacquired=True,
        all_reacquired_holds_acquired=True,
    ) == pytest.approx(1.0)
    assert _contact_anchor_pose_priority(
        phase=phase,
        contact_configuration_latched=True,
        anchor_individually_latched=True,
        all_individual_latches_acquired=True,
        anchor_reacquired=True,
        all_reacquired_holds_acquired=True,
    ) == pytest.approx(1.0)


def test_contact_centering_adds_clearance_correction_to_current_offset() -> None:
    hold = (1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0)
    normals = {4: (0.0, 1.0, 0.0), 9: (0.0, -1.0, 0.0)}

    first = _contact_centering_base_pose(
        hold,
        hold,
        mesh_clearance_m_by_anchor={4: 0.001, 9: 0.041},
        inward_normal_world_by_anchor=normals,
        max_offset_m=0.030,
    )
    assert first == pytest.approx((1.0, 1.98, 0.5, 0.0, 0.0, 0.0, 1.0))

    # Once the measured base has moved halfway, the remaining clearance
    # difference is added to that measured offset, so the terminal target
    # stays at the geometrically balanced -20 mm rather than collapsing to
    # a -10 mm steady-state error.
    second = _contact_centering_base_pose(
        hold,
        (1.0, 1.99, 0.5, 0.1, 0.0, 0.0, 0.995),
        mesh_clearance_m_by_anchor={4: 0.011, 9: 0.031},
        inward_normal_world_by_anchor=normals,
        max_offset_m=0.030,
    )
    assert second == pytest.approx(first)


def test_contact_centering_clamps_absolute_offset_and_fails_closed() -> None:
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert _contact_centering_base_pose(
        identity,
        identity,
        mesh_clearance_m_by_anchor={0: 0.0, 1: 0.20},
        inward_normal_world_by_anchor={
            0: (1.0, 0.0, 0.0),
            1: (-1.0, 0.0, 0.0),
        },
        max_offset_m=0.030,
    )[:3] == pytest.approx((-0.030, 0.0, 0.0))

    with pytest.raises(SchemaValidationError, match="exactly one normal"):
        _contact_centering_base_pose(
            identity,
            identity,
            mesh_clearance_m_by_anchor={0: 0.0, 1: 0.01},
            inward_normal_world_by_anchor={0: (1.0, 0.0, 0.0)},
            max_offset_m=0.030,
        )
    with pytest.raises(SchemaValidationError, match="opposing gripper"):
        _contact_centering_base_pose(
            identity,
            identity,
            mesh_clearance_m_by_anchor={0: 0.0, 1: 0.01},
            inward_normal_world_by_anchor={
                0: (1.0, 0.0, 0.0),
                1: (0.0, 1.0, 0.0),
            },
            max_offset_m=0.030,
        )


def test_post_first_arrest_centroidal_transfer_follows_unlatched_inward_axis() -> None:
    arrest = (1.0, 2.0, 0.5, 0.1, 0.2, 0.3, 0.9)

    target = _post_first_arrest_centroidal_transfer_pose(
        arrest,
        inward_normal_world=(0.0, 3.0, 0.0),
        maximum_transfer_m=0.03,
    )

    assert target[:3] == pytest.approx((1.0, 2.03, 0.5))
    assert target[3:] == arrest[3:]
    with pytest.raises(SchemaValidationError, match="finite and positive"):
        _post_first_arrest_centroidal_transfer_pose(
            arrest,
            inward_normal_world=(0.0, 1.0, 0.0),
            maximum_transfer_m=0.0,
        )


def test_contact_pair_recenters_on_near_surface_imbalance() -> None:
    assert _should_recenter_contact_pair(
        {0: 0.022, 1: 0.043},
        engagement_clearance_m=0.030,
        imbalance_clearance_m=0.003,
    )
    assert not _should_recenter_contact_pair(
        {0: 0.0435, 1: 0.0431},
        engagement_clearance_m=0.030,
        imbalance_clearance_m=0.003,
    )
    assert not _should_recenter_contact_pair(
        {0: 0.032, 1: 0.043},
        engagement_clearance_m=0.030,
        imbalance_clearance_m=0.003,
    )


def test_contact_anchor_speed_schedule_is_independent_per_mesh_surface() -> None:
    assert _contact_anchor_target_speed_limits_mps(
        base_limit_mps=0.010,
        mesh_clearance_m_by_anchor={0: 0.020, 1: 0.040},
        near_mesh_clearance_m=0.030,
        surface_arm_clearance_m=0.003,
        surface_creep_speed_limit_mps=0.001,
    ) == pytest.approx({0: 0.002, 1: 0.010})

    assert _contact_anchor_target_speed_limits_mps(
        base_limit_mps=0.010,
        mesh_clearance_m_by_anchor={0: 0.002, 1: 0.020},
        near_mesh_clearance_m=0.030,
        surface_arm_clearance_m=0.003,
        surface_creep_speed_limit_mps=0.001,
    ) == pytest.approx({0: 0.001, 1: 0.002})


def test_contact_anchor_speed_schedule_requires_the_opposing_pair() -> None:
    with pytest.raises(SchemaValidationError, match="exactly two"):
        _contact_anchor_target_speed_limits_mps(
            base_limit_mps=0.010,
            mesh_clearance_m_by_anchor={0: 0.020},
            near_mesh_clearance_m=0.030,
            surface_arm_clearance_m=0.003,
            surface_creep_speed_limit_mps=0.001,
        )


def test_post_first_arrest_accelerates_only_the_unlatched_anchor() -> None:
    assert _accelerate_unlatched_anchor_after_first_arrest(
        {0: 0.001, 1: 0.0002},
        latched_anchor_ids={0},
        maximum_speed_mps=0.010,
        creep_speed_mps=0.001,
        multiplier=3.0,
    ) == pytest.approx({0: 0.001, 1: 0.003})

    assert _accelerate_unlatched_anchor_after_first_arrest(
        {0: 0.001, 1: 0.002},
        latched_anchor_ids=set(),
        maximum_speed_mps=0.010,
        creep_speed_mps=0.001,
        multiplier=3.0,
    ) == pytest.approx({0: 0.001, 1: 0.002})


def test_contact_clearance_synchronization_slows_only_the_closer_side() -> None:
    assert _clearance_synchronized_contact_anchor_target_speed_limits_mps(
        {0: 0.002, 1: 0.010},
        mesh_clearance_m_by_anchor={0: 0.020, 1: 0.022},
        deadband_m=0.0005,
        full_slowdown_m=0.0015,
        minimum_speed_scale=0.05,
    ) == pytest.approx({0: 0.0001, 1: 0.010})

    assert _clearance_synchronized_contact_anchor_target_speed_limits_mps(
        {0: 0.002, 1: 0.002},
        mesh_clearance_m_by_anchor={0: 0.0200, 1: 0.0204},
        deadband_m=0.0005,
        full_slowdown_m=0.0015,
        minimum_speed_scale=0.05,
    ) == pytest.approx({0: 0.002, 1: 0.002})


def test_contact_clearance_synchronization_rejects_mismatched_pairs() -> None:
    with pytest.raises(SchemaValidationError, match="same opposing anchor pair"):
        _clearance_synchronized_contact_anchor_target_speed_limits_mps(
            {0: 0.002, 1: 0.002},
            mesh_clearance_m_by_anchor={0: 0.020, 2: 0.022},
            deadband_m=0.0005,
            full_slowdown_m=0.0015,
            minimum_speed_scale=0.05,
        )


def test_contact_pair_recenter_gate_fails_closed_on_invalid_geometry() -> None:
    with pytest.raises(SchemaValidationError, match="exactly two"):
        _should_recenter_contact_pair(
            {0: 0.001},
            engagement_clearance_m=0.030,
            imbalance_clearance_m=0.003,
        )


def test_contact_pair_centering_settle_uses_achieved_geometry_and_motion() -> None:
    common = {
        "speed_tolerance_mps": 0.0005,
        "imbalance_tolerance_m": 0.003,
        "max_tilt_rad": 0.020,
    }
    assert _contact_pair_centering_settled(
        {0: 0.0274, 1: 0.0271},
        base_linear_speed_mps=0.0002,
        measured_tilt_rad=0.0137,
        **common,
    )
    assert not _contact_pair_centering_settled(
        {0: 0.0274, 1: 0.0230},
        base_linear_speed_mps=0.0002,
        measured_tilt_rad=0.0137,
        **common,
    )
    assert not _contact_pair_centering_settled(
        {0: 0.0274, 1: 0.0271},
        base_linear_speed_mps=0.0006,
        measured_tilt_rad=0.0137,
        **common,
    )


def test_contact_pair_centering_settle_fails_closed_on_invalid_input() -> None:
    with pytest.raises(SchemaValidationError, match="exactly two"):
        _contact_pair_centering_settled(
            {0: 0.020},
            base_linear_speed_mps=0.0,
            speed_tolerance_mps=0.0005,
            imbalance_tolerance_m=0.003,
            measured_tilt_rad=0.0,
            max_tilt_rad=0.020,
        )
    with pytest.raises(SchemaValidationError, match="finite and positive"):
        _should_recenter_contact_pair(
            {0: 0.001, 1: 0.020},
            engagement_clearance_m=0.0,
            imbalance_clearance_m=0.003,
        )


def test_underactuated_centering_tilts_thrust_toward_world_y_target() -> None:
    hold = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)
    target = (0.0, -0.022, 0.5, 0.0, 0.0, 0.0, 1.0)

    tilted = _underactuated_contact_centering_pose(
        target,
        hold_pose=hold,
        current_pose=hold,
        current_linear_velocity_world=(0.0, 0.0, 0.0),
        speed_limit_mps=0.010,
        slowdown_distance_m=0.030,
        position_deadband_m=0.005,
        xy_p_gain=3.0,
        xy_d_gain=2.0,
        gravity_mps2=9.80665,
        max_tilt_rad=0.020,
    )
    _, rpy = pose_to_xyz_rpy(tilted)
    assert tilted[:3] == pytest.approx(target[:3])
    assert 0.0 < rpy[0] <= 0.020
    assert rpy[1:] == pytest.approx((0.0, 0.0))

    # Once position enters the settle deadband, the target returns to the
    # authored hold orientation so the simultaneous-contact gate cannot latch
    # a permanently tilted grasp.
    level = _underactuated_contact_centering_pose(
        (0.0, -0.003, 0.5, 0.0, 0.0, 0.0, 1.0),
        hold_pose=hold,
        current_pose=hold,
        current_linear_velocity_world=(0.0, 0.0, 0.0),
        speed_limit_mps=0.010,
        slowdown_distance_m=0.030,
        position_deadband_m=0.005,
        xy_p_gain=3.0,
        xy_d_gain=2.0,
        gravity_mps2=9.80665,
        max_tilt_rad=0.020,
    )
    assert level[3:] == pytest.approx(hold[3:])


def test_underactuated_centering_caps_tilt_and_validates_parameters() -> None:
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    tilted = _underactuated_contact_centering_pose(
        (1.0, 1.0, 0.0, *identity[3:]),
        hold_pose=identity,
        current_pose=identity,
        current_linear_velocity_world=(0.0, 0.0, 0.0),
        speed_limit_mps=0.010,
        slowdown_distance_m=0.030,
        position_deadband_m=0.001,
        xy_p_gain=100.0,
        xy_d_gain=2.0,
        gravity_mps2=9.80665,
        max_tilt_rad=0.020,
    )
    _, rpy = pose_to_xyz_rpy(tilted)
    assert math.hypot(rpy[0], rpy[1]) == pytest.approx(0.020)

    with pytest.raises(SchemaValidationError, match="finite and positive"):
        _underactuated_contact_centering_pose(
            identity,
            hold_pose=identity,
            current_pose=identity,
            current_linear_velocity_world=(0.0, 0.0, 0.0),
            speed_limit_mps=0.0,
            slowdown_distance_m=0.030,
            position_deadband_m=0.001,
            xy_p_gain=3.0,
            xy_d_gain=2.0,
            gravity_mps2=9.80665,
            max_tilt_rad=0.020,
        )


def test_all_selected_anchor_surface_load_settled_requires_two_sided_state() -> None:
    common = {
        "anchor_ids": (4, 9),
        "anchor_speed_threshold_mps": 0.020,
        "mesh_clearance_arm_threshold_m": 0.001,
        "mesh_clearance_m_by_anchor": {4: 0.0, 9: 0.0},
        "selected_joint_load_threshold_nm": 0.10,
        "selected_joint_load_nm_by_anchor": {4: 0.20, 9: 0.20},
    }

    assert _all_selected_anchor_surface_load_settled(
        **common,
        object_normal_relative_speed_mps_by_anchor={4: 0.001, 9: 0.019},
    )
    assert not _all_selected_anchor_surface_load_settled(
        **common,
        object_normal_relative_speed_mps_by_anchor={4: 0.001, 9: 0.021},
    )


def test_per_anchor_dock_load_does_not_cross_contaminate_grasp_branches() -> None:
    joint_ids = (
        "module_0:shared",
        "module_1:upstream",
        "module_1:terminal",
        "module_2:upstream",
        "module_2:terminal",
    )
    zero_row = (0.0,) * len(joint_ids)
    load_by_anchor, influential_by_anchor = _per_anchor_influential_dock_loads(
        (0, 1),
        ordered_joint_ids=joint_ids,
        anchor_jacobians={
            0: (
                (1.0, 0.5, 0.0, 0.0, 0.0),
                zero_row,
                zero_row,
                (0.0, 0.0, 1.0, 0.0, 0.0),
                zero_row,
                zero_row,
            ),
            1: (
                (1.0, 0.0, 0.0, 0.5, 0.0),
                zero_row,
                zero_row,
                (0.0, 0.0, 0.0, 0.0, 1.0),
                zero_row,
                zero_row,
            ),
        },
        applied_joint_load_nm={
            "module_0:shared": 0.10,
            "module_1:upstream": 0.80,
            "module_1:terminal": 0.05,
            "module_2:upstream": 0.20,
            "module_2:terminal": 0.04,
        },
        required_joint_id_by_anchor={
            0: "module_1:terminal",
            1: "module_2:terminal",
        },
    )

    assert load_by_anchor == pytest.approx({0: 0.80, 1: 0.20})
    assert influential_by_anchor == {
        0: (
            "module_0:shared",
            "module_1:upstream",
            "module_1:terminal",
        ),
        1: (
            "module_0:shared",
            "module_2:upstream",
            "module_2:terminal",
        ),
    }


def test_selected_anchor_surface_load_candidates_support_independent_dwell() -> None:
    assert _selected_anchor_surface_load_settle_candidates(
        (4, 9),
        object_normal_relative_speed_mps_by_anchor={4: 0.001, 9: 0.004},
        mesh_clearance_m_by_anchor={4: 0.0, 9: 0.0},
        selected_joint_load_nm_by_anchor={4: 0.20, 9: 0.20},
        anchor_speed_threshold_mps=0.003,
        mesh_clearance_arm_threshold_m=0.001,
        selected_joint_load_threshold_nm=0.10,
    ) == {4: True, 9: False}

    assert _selected_anchor_surface_load_settle_candidates(
        (4, 9),
        object_normal_relative_speed_mps_by_anchor={4: 0.001, 9: 0.001},
        mesh_clearance_m_by_anchor={4: 0.002, 9: 0.0},
        selected_joint_load_nm_by_anchor={4: 0.20, 9: 0.20},
        anchor_speed_threshold_mps=0.003,
        mesh_clearance_arm_threshold_m=0.001,
        selected_joint_load_threshold_nm=0.10,
    ) == {4: False, 9: True}

    assert _selected_anchor_surface_load_settle_candidates(
        (4, 9),
        object_normal_relative_speed_mps_by_anchor={4: 0.001, 9: 0.001},
        mesh_clearance_m_by_anchor={4: 0.0, 9: 0.0},
        selected_joint_load_nm_by_anchor={4: 0.09, 9: 0.10},
        anchor_speed_threshold_mps=0.003,
        mesh_clearance_arm_threshold_m=0.001,
        selected_joint_load_threshold_nm=0.10,
    ) == {4: False, 9: True}


def test_surface_load_arrest_candidates_do_not_wait_for_velocity_settle() -> None:
    common = {
        "anchor_ids": (4, 9),
        "mesh_clearance_m_by_anchor": {4: 0.0005, 9: 0.0008},
        "mesh_clearance_arm_threshold_m": 0.001,
        "selected_joint_load_threshold_nm": 0.10,
    }

    assert _selected_anchor_surface_load_arrest_candidates(
        **common,
        selected_joint_load_nm_by_anchor={4: 0.20, 9: 0.15},
    ) == {4: True, 9: True}
    assert _selected_anchor_surface_load_arrest_candidates(
        **common,
        selected_joint_load_nm_by_anchor={4: 0.20, 9: 0.09},
    ) == {4: True, 9: False}
    assert _selected_anchor_surface_load_arrest_candidates(
        **{
            **common,
            "mesh_clearance_m_by_anchor": {4: 0.0011, 9: 0.0008},
        },
        selected_joint_load_nm_by_anchor={4: 0.20, 9: 0.15},
    ) == {4: False, 9: True}


def test_all_selected_anchor_surface_load_settled_fails_closed_on_bad_coverage() -> (
    None
):
    with pytest.raises(SchemaValidationError, match="cover exactly"):
        _all_selected_anchor_surface_load_settled(
            (4, 9),
            object_normal_relative_speed_mps_by_anchor={4: 0.0},
            mesh_clearance_m_by_anchor={4: 0.0, 9: 0.0},
            selected_joint_load_nm_by_anchor={4: 0.2, 9: 0.2},
            anchor_speed_threshold_mps=0.020,
            mesh_clearance_arm_threshold_m=0.001,
            selected_joint_load_threshold_nm=0.10,
        )
    with pytest.raises(SchemaValidationError, match="unique anchor"):
        _all_selected_anchor_surface_load_settled(
            (4, 4),
            object_normal_relative_speed_mps_by_anchor={4: 0.0},
            mesh_clearance_m_by_anchor={4: 0.0},
            selected_joint_load_nm_by_anchor={4: 0.2},
            anchor_speed_threshold_mps=0.020,
            mesh_clearance_arm_threshold_m=0.001,
            selected_joint_load_threshold_nm=0.10,
        )


def test_selected_gripper_proxy_pad_fits_sampled_outer_face_and_clears_mesh() -> None:
    surfaces = tuple(
        SimpleNamespace(
            module_id=module_id,
            mechanism_link_id=f"dock_{module_id}",
            port_local_id=f"connect_{module_id}",
        )
        for module_id in (1, 2)
    )
    physical_model = SimpleNamespace(
        joints=[
            SimpleNamespace(
                joint_id=f"connect_{module_id}",
                parent_link=f"dock_{module_id}",
                origin_xyz=(0.115, 0.0, 0.0),
                origin_rpy=(0.0, 0.0, 0.0),
            )
            for module_id in (1, 2)
        ]
    )
    surface_samples = tuple(
        (0.100, y_value, z_value)
        for y_value in (-0.020, 0.0, 0.020)
        for z_value in (-0.020, 0.0, 0.020)
    )
    local_meshes = tuple(
        _SelectedMeshLocalAABB(
            module_id=module_id,
            link_id=f"dock_{module_id}",
            primitive_id=f"dock_{module_id}:collision:0",
            geometry_ref=f"dock_{module_id}.stl",
            minimum_local=(-0.020, -0.020, -0.020),
            maximum_local=(0.100, 0.020, 0.020),
            surface_sample_points_local=surface_samples,
        )
        for module_id in (1, 2)
    )

    specs = _selected_gripper_proxy_pad_specs(
        surfaces,
        local_meshes,
        physical_model,
    )

    assert [spec.module_id for spec in specs] == [1, 2]
    for spec in specs:
        assert spec.center_local == pytest.approx((0.102, 0.0, 0.0))
        assert spec.orientation_local_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
        assert spec.size_m == pytest.approx((0.002, 0.030, 0.030))
        assert spec.mesh_surface_projection_m == pytest.approx(0.100)
        assert spec.inner_face_projection_m == pytest.approx(0.101)
        assert spec.outer_face_projection_m == pytest.approx(0.103)
        assert spec.tangential_surface_span_m == pytest.approx((0.040, 0.040))
        assert (
            spec.outer_face_projection_m - spec.mesh_surface_projection_m
        ) > 0.002


def test_selected_gripper_cone_proxy_uses_approved_tiles_on_selected_modules() -> None:
    surfaces = (
        SimpleNamespace(
            module_id=1,
            mechanism_link_id="yaw_dock_mech2",
        ),
        SimpleNamespace(
            module_id=2,
            mechanism_link_id="yaw_dock_mech1",
        ),
    )

    specs = _selected_gripper_cone_proxy_pad_specs(
        surfaces,
        urdf_path="assets/robots/holon/holon.urdf",
    )

    assert len(specs) == 150
    assert sum(spec.module_id == 1 for spec in specs) == 75
    assert sum(spec.module_id == 2 for spec in specs) == 75
    assert {spec.link_id for spec in specs} == {
        "yaw_dock_mech1",
        "yaw_dock_mech2",
    }
    assert all(spec.size_m[2] == pytest.approx(0.0008) for spec in specs)
    assert all(spec.inner_face_surface_gap_m == pytest.approx(0.0002) for spec in specs)

    surface_meshes = _cone_proxy_pad_surface_local_meshes(specs)
    assert len(surface_meshes) == 150
    assert all(len(mesh.surface_sample_points_local) == 14 for mesh in surface_meshes)
    assert {
        (mesh.module_id, mesh.link_id) for mesh in surface_meshes
    } == {
        (1, "yaw_dock_mech2"),
        (2, "yaw_dock_mech1"),
    }
    assert all(
        mesh.geometry_ref == "diagnostic_cone_proxy_box"
        for mesh in surface_meshes
    )


def test_selected_gripper_proxy_pad_rejects_too_small_outer_face() -> None:
    surfaces = (
        SimpleNamespace(
            module_id=1,
            mechanism_link_id="dock_1",
            port_local_id="connect_1",
        ),
        SimpleNamespace(
            module_id=2,
            mechanism_link_id="dock_2",
            port_local_id="connect_2",
        ),
    )
    physical_model = SimpleNamespace(
        joints=[
            SimpleNamespace(
                joint_id=f"connect_{module_id}",
                parent_link=f"dock_{module_id}",
                origin_xyz=(0.115, 0.0, 0.0),
                origin_rpy=(0.0, 0.0, 0.0),
            )
            for module_id in (1, 2)
        ]
    )
    local_meshes = tuple(
        _SelectedMeshLocalAABB(
            module_id=module_id,
            link_id=f"dock_{module_id}",
            primitive_id=f"dock_{module_id}:collision:0",
            geometry_ref=f"dock_{module_id}.stl",
            minimum_local=(0.100, -0.005, -0.005),
            maximum_local=(0.100, 0.005, 0.005),
            surface_sample_points_local=(
                (0.100, -0.005, -0.005),
                (0.100, 0.005, 0.005),
            ),
        )
        for module_id in (1, 2)
    )

    with pytest.raises(SchemaValidationError, match="does not fit"):
        _selected_gripper_proxy_pad_specs(
            surfaces,
            local_meshes,
            physical_model,
        )


def test_selected_gripper_local_aabb_uses_urdf_mesh_scale_and_collision_pose(
    tmp_path,
) -> None:
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    mesh_path = mesh_dir / "dock.STL"
    mesh_path.write_text(
        """solid dock
facet normal 0 0 1
outer loop
vertex 0 0 -0.5
vertex 1 0 0.5
vertex 0 2 0.5
endloop
endfacet
endsolid dock
""",
        encoding="ascii",
    )
    urdf_path = tmp_path / "module.urdf"
    urdf_path.write_text(
        """<robot name="module">
  <link name="dock_mech">
    <collision>
      <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
      <geometry><mesh filename="mesh/dock.STL" scale="2 1 1"/></geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    surface = SimpleNamespace(
        module_id=2,
        mechanism_link_id="dock_mech",
        collision_primitives=(
            SimpleNamespace(
                primitive_id="dock_mech:collision:0",
                primitive_type="mesh",
                geometry_ref="mesh/dock.STL",
            ),
        ),
    )

    bounds = _selected_gripper_mesh_local_aabbs(
        (surface,),
        urdf_path=urdf_path,
    )[0]

    assert bounds.module_id == 2
    assert bounds.link_id == "dock_mech"
    assert bounds.minimum_local == pytest.approx((-1.0, 0.0, -0.5))
    assert bounds.maximum_local == pytest.approx((1.0, 2.0, 0.5))
    assert len(bounds.surface_sample_points_local) == 4
    assert bounds.surface_sample_points_local[0] == pytest.approx((1.0, 0.0, -0.5))


def test_gripper_surface_samples_exclude_empty_aabb_volume() -> None:
    bounds = _SelectedMeshLocalAABB(
        module_id=2,
        link_id="dock_mech",
        primitive_id="dock_mech:collision:0",
        geometry_ref="dock.STL",
        minimum_local=(-2.0, -2.0, -2.0),
        maximum_local=(2.0, 2.0, 2.0),
        surface_sample_points_local=((2.0, 0.0, 0.0),),
    )
    body_poses = {(2, "dock_mech"): (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)}

    assert _gripper_object_clearance_from_body_poses(
        (bounds,),
        body_poses,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        [1.0, 1.0, 1.0],
    ) == pytest.approx(0.0)
    assert _gripper_object_surface_sample_clearance_from_body_poses(
        (bounds,),
        body_poses,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        [1.0, 1.0, 1.0],
    ) == pytest.approx(1.5)

    (
        clearance,
        application_point,
        surface_normal_world,
    ) = _gripper_object_surface_sample_query_from_body_poses(
        (bounds,),
        body_poses,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        [1.0, 1.0, 1.0],
    )
    assert clearance == pytest.approx(1.5)
    assert application_point == pytest.approx((2.0, 0.0, 0.0))
    assert surface_normal_world == pytest.approx((1.0, 0.0, 0.0))


def test_gripper_surface_query_selects_least_penetrating_authored_sample() -> None:
    bounds = _SelectedMeshLocalAABB(
        module_id=2,
        link_id="dock_mech",
        primitive_id="dock_mech:collision:0",
        geometry_ref="dock.STL",
        minimum_local=(-0.4, 0.0, 0.0),
        maximum_local=(0.49, 0.0, 0.0),
        surface_sample_points_local=(
            (0.0, 0.0, 0.0),
            (0.49, 0.0, 0.0),
            (-0.4, 0.0, 0.0),
        ),
    )

    (
        clearance,
        application_point,
        surface_normal_world,
    ) = _gripper_object_surface_sample_query_from_body_poses(
        (bounds,),
        {(2, "dock_mech"): (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)},
        (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
        [1.0, 1.0, 1.0],
    )

    assert clearance == pytest.approx(0.0)
    assert application_point == pytest.approx((1.49, 2.0, 3.0))
    assert surface_normal_world == pytest.approx((1.0, 0.0, 0.0))


def test_gripper_clearance_transforms_link_local_mesh_aabb_at_each_body_pose() -> None:
    half_sqrt_two = math.sqrt(0.5)
    bounds = _SelectedMeshLocalAABB(
        module_id=2,
        link_id="dock_mech",
        primitive_id="dock_mech:collision:0",
        geometry_ref="dock.STL",
        minimum_local=(-1.0, 0.0, -0.5),
        maximum_local=(1.0, 2.0, 0.5),
    )

    clearance = _gripper_object_clearance_from_body_poses(
        (bounds,),
        {
            (2, "dock_mech"): (
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                half_sqrt_two,
                half_sqrt_two,
            )
        },
        (7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        [1.0, 1.0, 1.0],
    )

    # Link-local bounds become world x=[8, 10], while the object is x=[6.5, 7.5].
    assert clearance == pytest.approx(0.5)


def test_axial_overlap_uses_least_selected_mesh_projection() -> None:
    first = _SelectedMeshLocalAABB(
        module_id=1,
        link_id="first_dock_mech",
        primitive_id="first:collision:0",
        geometry_ref="first.STL",
        minimum_local=(-0.5, -0.5, -0.5),
        maximum_local=(0.5, 0.5, 0.5),
    )
    second = _SelectedMeshLocalAABB(
        module_id=2,
        link_id="second_dock_mech",
        primitive_id="second:collision:0",
        geometry_ref="second.STL",
        minimum_local=(-0.5, -0.5, -0.5),
        maximum_local=(0.5, 0.5, 0.5),
    )
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    overlap = _minimum_gripper_object_axial_overlap_from_body_poses(
        (first, second),
        {
            (1, "first_dock_mech"): identity,
            (2, "second_dock_mech"): (
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        },
        (0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        [1.0, 1.0, 1.0],
        axis_world=(1.0, 0.0, 0.0),
    )

    # Object x=[0.25, 1.25].  The first mesh reaches x=0.5 (0.25 m
    # overlap), while the second reaches x=0.75 (0.50 m overlap).
    assert overlap == pytest.approx(0.25)


def test_mesh_aware_staging_retreats_actual_mesh_to_required_clearance() -> None:
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    bounds = _SelectedMeshLocalAABB(
        module_id=2,
        link_id="dock_mech",
        primitive_id="dock_mech:collision:0",
        geometry_ref="dock.STL",
        minimum_local=(-0.5, -0.5, -0.5),
        maximum_local=(0.5, 0.5, 0.5),
    )

    plan = _mesh_aware_staging_plan(
        (bounds,),
        {(2, "dock_mech"): identity},
        grasp_base_pose=identity,
        object_pose=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        object_size=[1.0, 1.0, 1.0],
        required_clearance_m=0.2,
        maximum_retreat_m=0.5,
    )

    assert plan.retreat_distance_m == pytest.approx(0.2)
    assert plan.predicted_clearance_m == pytest.approx(0.2)
    assert plan.base_pose_world[:3] == pytest.approx((-0.2, 0.0, 0.0))
    assert plan.approach_axis_world == pytest.approx((1.0, 0.0, 0.0))


def test_mesh_aware_staging_fails_closed_when_standoff_is_insufficient() -> None:
    bounds = _SelectedMeshLocalAABB(
        module_id=1,
        link_id="dock_mech",
        primitive_id="dock_mech:collision:0",
        geometry_ref="dock.STL",
        minimum_local=(-0.5, -0.5, -0.5),
        maximum_local=(0.5, 0.5, 0.5),
    )
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    with pytest.raises(SchemaValidationError, match="cannot reach"):
        _mesh_aware_staging_plan(
            (bounds,),
            {(1, "dock_mech"): identity},
            grasp_base_pose=identity,
            object_pose=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            object_size=[1.0, 1.0, 1.0],
            required_clearance_m=0.2,
            maximum_retreat_m=0.1,
        )


def test_mesh_aware_anchor_opening_moves_selected_mesh_outward_in_base_frame() -> None:
    bounds = _SelectedMeshLocalAABB(
        module_id=1,
        link_id="dock_mech",
        primitive_id="dock_mech:collision:0",
        geometry_ref="dock.STL",
        minimum_local=(-0.5, -0.5, -0.5),
        maximum_local=(0.5, 0.5, 0.5),
    )
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    plan = _mesh_aware_anchor_opening_plan(
        (bounds,),
        {(1, "dock_mech"): identity},
        anchor_id_by_module_link={(1, "dock_mech"): 7},
        anchor_pose_world_by_id={7: identity},
        inward_normal_world_by_anchor={7: (1.0, 0.0, 0.0)},
        grasp_base_pose=identity,
        object_pose=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        object_size=[1.0, 1.0, 1.0],
        required_clearance_m=0.2,
        maximum_opening_m=0.5,
    )

    assert plan.anchor_poses_base[7][:3] == pytest.approx((-0.2, 0.0, 0.0))
    assert plan.outward_distance_m_by_anchor[7] == pytest.approx(0.2)
    assert plan.predicted_clearance_m_by_anchor[7] == pytest.approx(0.2)


def test_base_phase_targets_separate_mesh_staging_from_contact_approach() -> None:
    hover = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    approach = (0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)
    grasp = (0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)
    common = {
        "hover_base_pose": hover,
        "approach_base_pose": approach,
        "grasp_base_pose": grasp,
        "lift_base_pose": (0.5, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
        "transport_base_pose": (0.7, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
        "place_base_pose": (0.7, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        "retreat_base_pose": (0.6, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
    }

    assert (
        _base_target_for_phase(Order8NaturalContactPhase.APPROACH, **common) == approach
    )
    assert (
        _base_target_for_phase(Order8NaturalContactPhase.CONTACT_ACQUISITION, **common)
        == grasp
    )


def test_floor_clear_grasp_base_uses_tangential_region_not_floor_penetration() -> None:
    plan = _floor_clear_grasp_base_plan(
        floor_base_pose=(0.0, 0.0, 0.143, 0.0, 0.0, 0.0, 1.0),
        unconstrained_grasp_base_pose=(
            0.5,
            0.0,
            0.122,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        inward_normal_world_by_anchor={
            0: (0.0, -1.0, 0.0),
            1: (0.0, 1.0, 0.0),
        },
        tangential_tolerance_m=0.05,
        additional_floor_clearance_m=0.01,
    )

    assert plan.base_pose_world == pytest.approx((0.5, 0.0, 0.153, 0.0, 0.0, 0.0, 1.0))
    assert plan.vertical_correction_m == pytest.approx(0.031)
    assert plan.normal_correction_m_by_anchor == pytest.approx({0: 0.0, 1: 0.0})
    assert all(
        max(abs(value) for value in correction) == pytest.approx(0.031)
        for correction in plan.tangential_correction_m_by_anchor.values()
    )


def test_floor_clear_grasp_base_preserves_an_already_airborne_target() -> None:
    plan = _floor_clear_grasp_base_plan(
        floor_base_pose=(0.0, 0.0, 0.143, 0.0, 0.0, 0.0, 1.0),
        unconstrained_grasp_base_pose=(
            0.5,
            0.0,
            0.20,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        inward_normal_world_by_anchor={0: (0.0, 1.0, 0.0)},
        tangential_tolerance_m=0.05,
    )

    assert plan.base_pose_world == plan.unconstrained_base_pose_world
    assert plan.vertical_correction_m == 0.0


def test_floor_clear_grasp_base_rejects_normal_or_out_of_region_correction() -> None:
    common = {
        "floor_base_pose": (0.0, 0.0, 0.143, 0.0, 0.0, 0.0, 1.0),
        "unconstrained_grasp_base_pose": (
            0.5,
            0.0,
            0.122,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        "tangential_tolerance_m": 0.05,
    }
    with pytest.raises(SchemaValidationError, match="surface normal"):
        _floor_clear_grasp_base_plan(
            **common,
            inward_normal_world_by_anchor={0: (0.0, 0.0, 1.0)},
            additional_floor_clearance_m=0.0,
        )
    with pytest.raises(SchemaValidationError, match="tangential contact region"):
        _floor_clear_grasp_base_plan(
            floor_base_pose=(0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 1.0),
            unconstrained_grasp_base_pose=common["unconstrained_grasp_base_pose"],
            inward_normal_world_by_anchor={0: (0.0, 1.0, 0.0)},
            tangential_tolerance_m=0.05,
            additional_floor_clearance_m=0.0,
        )


def test_base_hold_settle_gate_requires_pose_and_low_speed() -> None:
    target = (1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)

    assert _base_hold_settled(
        target,
        (0.95, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        base_linear_speed_mps=0.002,
        position_tolerance_m=0.08,
        speed_tolerance_mps=0.003,
    )
    assert not _base_hold_settled(
        target,
        (0.95, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        base_linear_speed_mps=0.004,
        position_tolerance_m=0.08,
        speed_tolerance_mps=0.003,
    )
    assert not _base_hold_settled(
        target,
        (0.90, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        base_linear_speed_mps=0.002,
        position_tolerance_m=0.08,
        speed_tolerance_mps=0.003,
    )


def test_contact_anchor_overtravel_moves_both_targets_toward_object_center() -> None:
    selections = [
        SimpleNamespace(anchor_id=0, inward_normal_world=(0.0, 1.0, 0.0)),
        SimpleNamespace(anchor_id=1, inward_normal_world=(0.0, -1.0, 0.0)),
    ]

    targets = _desired_anchor_poses(
        selections,
        (1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        [0.3, 0.4, 0.15],
        pregrasp=False,
        inward_overtravel_m=0.06,
    )

    assert targets[0][:3] == pytest.approx((1.0, 1.86, 0.5))
    assert targets[1][:3] == pytest.approx((1.0, 2.14, 0.5))


def test_contact_anchor_speed_schedule_slows_only_near_mesh_boundary() -> None:
    assert _contact_anchor_target_speed_limit_mps(
        base_limit_mps=0.01,
        mesh_clearance_m=0.005,
        near_mesh_clearance_m=0.004,
        surface_arm_clearance_m=0.001,
        surface_creep_speed_limit_mps=0.0005,
    ) == pytest.approx(0.01)
    assert _contact_anchor_target_speed_limit_mps(
        base_limit_mps=0.01,
        mesh_clearance_m=0.004,
        near_mesh_clearance_m=0.004,
        surface_arm_clearance_m=0.001,
        surface_creep_speed_limit_mps=0.0005,
    ) == pytest.approx(0.002)
    assert _contact_anchor_target_speed_limit_mps(
        base_limit_mps=0.01,
        mesh_clearance_m=0.001,
        near_mesh_clearance_m=0.004,
        surface_arm_clearance_m=0.001,
        surface_creep_speed_limit_mps=0.0005,
    ) == pytest.approx(0.0005)
