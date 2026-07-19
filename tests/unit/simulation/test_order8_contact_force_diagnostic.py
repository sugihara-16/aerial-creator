from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.order8_contact_force_diagnostic import (
    _build_qclose_fixture_from_state_trace,
    _fast_config,
    _load_fixed_closure_velocity_targets,
    _load_near_contact_fixture_report,
    _source_probe_option,
    _supports_phase_continuation,
    _transform_near_contact_fixture,
    build_parser,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)


@dataclass(frozen=True)
class _Config:
    base_translation_speed_limit_mps: float = 0.10
    contact_base_translation_speed_limit_mps: float = 0.01
    hover_dwell_s: float = 2.0
    anchor_translation_speed_limit_mps: float = 0.01
    contact_surface_creep_speed_limit_mps: float = 0.001
    contact_dwell_s: float = 0.25
    payload_load_transfer_s: float = 1.0
    contact_force_ramp_s: float = 40.0
    contact_stall_dwell_s: float = 0.10
    contact_acquisition_timeout_s: float = 90.0
    object_mass_kg: float = 1.0
    object_friction: float = 0.6
    selected_gripper_friction: float = 2.0
    selected_gripper_compliant_contact_stiffness_n_per_m: float = 7500.0
    selected_gripper_compliant_contact_damping_n_s_per_m: float = 75.0
    normal_force_target_per_contact_n: float = 11.0
    contact_stall_anchor_speed_threshold_mps: float = 0.0015
    max_tangential_slip_speed_mps: float = 0.02
    max_contact_point_slip_displacement_m: float = 0.03


def test_full_sequence_preserves_nominal_contact_dwell() -> None:
    config = _fast_config(
        _Config(),
        speed_scale=2.0,
        force_ramp_s=40.0,
        object_width_padding_m=0.0,
        full_sequence=True,
        object_friction=None,
        contact_dwell_s=None,
    )

    assert config.contact_dwell_s == 0.25


def test_minimum_fixture_uses_short_contact_dwell() -> None:
    config = _fast_config(
        _Config(),
        speed_scale=2.0,
        force_ramp_s=4.0,
        object_width_padding_m=0.0,
        full_sequence=False,
        object_friction=None,
        contact_dwell_s=None,
    )

    assert config.contact_dwell_s == 0.05


def test_fixture_can_retain_production_contact_dwell_explicitly() -> None:
    config = _fast_config(
        _Config(),
        speed_scale=2.0,
        force_ramp_s=40.0,
        object_width_padding_m=0.0,
        full_sequence=False,
        object_friction=None,
        contact_dwell_s=0.25,
    )

    assert config.contact_dwell_s == 0.25
    assert config.contact_acquisition_timeout_s == 60.0


def test_parser_exposes_acceptance_ineligible_phase_continuation() -> None:
    args = build_parser().parse_args(
        [
            "--precontact-fixture-report",
            "fixture.json",
            "--continue-after-force-ramp",
        ]
    )

    assert args.continue_after_force_ramp is True


def test_parser_exposes_separated_lift_transition() -> None:
    args = build_parser().parse_args(
        [
            "--continue-after-force-ramp",
            "--separate-lift-transition",
            "--lift-bias-delay-s",
            "1.5",
            "--disable-payload-feedforward",
        ]
    )

    assert args.separate_lift_transition is True
    assert args.lift_bias_delay_s == pytest.approx(1.5)
    assert args.disable_payload_feedforward is True


def test_parser_exposes_payload_coupling_component_isolation() -> None:
    args = build_parser().parse_args(
        [
            "--continue-after-force-ramp",
            "--separate-lift-transition",
            "--payload-coupling-component-mode",
            "translational_force_only",
        ]
    )

    assert args.payload_coupling_component_mode == "translational_force_only"


def test_parser_exposes_acceptance_ineligible_proxy_pad_and_friction() -> None:
    args = build_parser().parse_args(
        [
            "--proxy-pad",
            "--selected-gripper-friction",
            "3.0",
            "--payload-load-transfer-s",
            "0.5",
        ]
    )

    assert args.proxy_pad is True
    assert args.selected_gripper_friction == pytest.approx(3.0)
    assert args.payload_load_transfer_s == pytest.approx(0.5)


def test_parser_exposes_approved_cone_proxy_pad() -> None:
    args = build_parser().parse_args(["--cone-proxy-pad"])

    assert args.cone_proxy_pad is True
    assert args.proxy_pad is False


def test_parser_exposes_diagnostic_contact_closure_joint_speed() -> None:
    args = build_parser().parse_args(
        ["--contact-closure-joint-speed-radps", "0.005"]
    )

    assert args.contact_closure_joint_speed_radps == pytest.approx(0.005)


def test_parser_and_fast_config_expose_diagnostic_object_mass_override() -> None:
    args = build_parser().parse_args(["--object-mass-kg", "0.9"])
    source = _Config()
    config = _fast_config(
        source,
        speed_scale=2.0,
        force_ramp_s=4.0,
        object_width_padding_m=0.0,
        full_sequence=False,
        object_mass_kg=args.object_mass_kg,
        object_friction=None,
        contact_dwell_s=None,
    )

    assert args.object_mass_kg == pytest.approx(0.9)
    assert config.object_mass_kg == pytest.approx(0.9)
    assert source.object_mass_kg == pytest.approx(1.0)


def test_parser_exposes_selected_mesh_contact_compliance_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--selected-gripper-compliant-contact-stiffness",
            "6000",
            "--selected-gripper-compliant-contact-damping",
            "60",
        ]
    )

    assert args.selected_gripper_compliant_contact_stiffness == pytest.approx(6000.0)
    assert args.selected_gripper_compliant_contact_damping == pytest.approx(60.0)


def test_parser_exposes_post_grasp_joint_torque_bias() -> None:
    args = build_parser().parse_args(
        ["--post-grasp-joint-torque-bias-nm", "0.5"]
    )

    assert args.post_grasp_joint_torque_bias_nm == pytest.approx(0.5)


def test_parser_exposes_diagnostic_slip_speed_safe_hold_disable() -> None:
    args = build_parser().parse_args(["--disable-slip-speed-safe-hold"])

    assert args.disable_slip_speed_safe_hold is True


def test_parser_exposes_diagnostic_all_safe_hold_disable() -> None:
    args = build_parser().parse_args(["--disable-all-safe-hold"])

    assert args.disable_all_safe_hold is True


def test_parser_exposes_diagnostic_object_rotation_lock() -> None:
    args = build_parser().parse_args(["--lock-object-rotation"])

    assert args.lock_object_rotation is True


def test_parser_exposes_diagnostic_anchor_hold_joint_correction() -> None:
    args = build_parser().parse_args(["--anchor-hold-joint-correction"])

    assert args.anchor_hold_joint_correction is True


def test_parser_exposes_diagnostic_loaded_state_rebase() -> None:
    args = build_parser().parse_args(
        [
            "--continue-after-force-ramp",
            "--separate-lift-transition",
            "--loaded-state-rebase",
        ]
    )

    assert args.loaded_state_rebase is True


def test_fast_config_can_override_shared_payload_progress_duration() -> None:
    source = _Config()
    config = _fast_config(
        source,
        speed_scale=2.0,
        force_ramp_s=4.0,
        object_width_padding_m=0.0,
        full_sequence=False,
        object_friction=None,
        contact_dwell_s=None,
        payload_load_transfer_s=0.5,
    )

    assert config.payload_load_transfer_s == pytest.approx(0.5)
    assert source.payload_load_transfer_s == pytest.approx(1.0)


def test_fast_config_can_override_selected_mesh_contact_compliance() -> None:
    source = _Config()
    config = _fast_config(
        source,
        speed_scale=2.0,
        force_ramp_s=4.0,
        object_width_padding_m=0.0,
        full_sequence=False,
        object_friction=None,
        contact_dwell_s=None,
        selected_gripper_compliant_contact_stiffness=6000.0,
        selected_gripper_compliant_contact_damping=60.0,
    )

    assert (
        config.selected_gripper_compliant_contact_stiffness_n_per_m
        == pytest.approx(6000.0)
    )
    assert (
        config.selected_gripper_compliant_contact_damping_n_s_per_m
        == pytest.approx(60.0)
    )
    assert (
        source.selected_gripper_compliant_contact_stiffness_n_per_m
        == pytest.approx(7500.0)
    )
    assert (
        source.selected_gripper_compliant_contact_damping_n_s_per_m
        == pytest.approx(75.0)
    )


def test_parser_exposes_qclose_state_trace_fixture() -> None:
    args = build_parser().parse_args(
        ["--qclose-fixture-state-trace", "/tmp/grasp.json"]
    )

    assert args.qclose_fixture_state_trace == "/tmp/grasp.json"


def test_recorded_probe_option_rejects_missing_or_conflicting_values() -> None:
    assert _source_probe_option(["--config", "one.yaml"], "--config") == "one.yaml"
    with pytest.raises(ValueError, match="has no --config"):
        _source_probe_option([], "--config")
    with pytest.raises(ValueError, match="conflicting"):
        _source_probe_option(
            ["--config", "one.yaml", "--config", "two.yaml"],
            "--config",
        )


def test_lift_trace_builds_complete_measured_qclose_checkpoint() -> None:
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    morphology = build_representative_order8_morphology(physical_model)
    local_dock_joint_ids = sorted(
        {
            str(port.mechanical_limits["mechanism_joint_id"])
            for port in physical_model.dock_ports
        }
    )
    module_ids = sorted(module.module_id for module in morphology.modules)

    def frame(time_s: float, phase: str) -> dict[str, object]:
        return {
            "simulation_time_s": time_s,
            "phase": phase,
            "modules": {
                str(module_id): {
                    "root_pose_world": [
                        0.1 * module_id,
                        -0.2 * module_id,
                        0.4,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "root_twist_world": [0.01 * module_id] * 6,
                    "joint_positions_rad": [0.0] * len(local_dock_joint_ids),
                    "joint_velocities_radps": [0.001 * (module_id + 1)]
                    * len(local_dock_joint_ids),
                }
                for module_id in module_ids
            },
            "object_pose_world": [0.8, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            "object_twist_world": [0.0] * 6,
        }

    trace = {
        "graph_hash": morphology.stable_hash(),
        "source_urdf_sha256": "source-hash",
        "trace_payload_hash": "trace-hash",
        "joint_names_by_module": {
            str(module_id): list(local_dock_joint_ids) for module_id in module_ids
        },
        "frames": [frame(0.0, "contact_acquisition"), frame(0.1, "lift")],
    }

    fixture = _build_qclose_fixture_from_state_trace(
        trace,
        morphology=morphology,
        physical_model=physical_model,
        resolved_urdf_path=Path(physical_model.urdf_path),
    )
    state = fixture["order8_natural_contact_qclose_checkpoint_state"]

    assert isinstance(state, dict)
    assert state["schema_version"] == "order8_qclose_checkpoint_state_v1"
    assert set(state["module_root_poses"]) == {"0", "1", "2"}
    assert len(state["joint_positions_rad"]) == 3 * len(local_dock_joint_ids)
    assert set(state["anchor_hold_poses_base"]) == {"0", "1"}
    assert fixture[
        "order8_natural_contact_qclose_trace_fixture_provenance"
    ]["source_frame_simulation_time_s"] == pytest.approx(0.1)


def test_phase_continuation_accepts_precontact_or_exact_qclose_only() -> None:
    assert _supports_phase_continuation(
        precontact_base_pose=[0.0] * 7,
        qclose_base_pose=None,
        qclose_checkpoint_state=None,
    )
    assert _supports_phase_continuation(
        precontact_base_pose=None,
        qclose_base_pose=[0.0] * 7,
        qclose_checkpoint_state={"schema_version": "order8_qclose_checkpoint_state_v1"},
    )
    assert not _supports_phase_continuation(
        precontact_base_pose=None,
        qclose_base_pose=[0.0] * 7,
        qclose_checkpoint_state=None,
    )


def test_phase_continuation_accepts_collision_free_near_contact_state() -> None:
    assert _supports_phase_continuation(
        precontact_base_pose=None,
        near_contact_base_pose=[0.0] * 7,
        qclose_base_pose=None,
        qclose_checkpoint_state=None,
    )


def test_parser_exposes_near_contact_fixture_report() -> None:
    args = build_parser().parse_args(
        ["--near-contact-fixture-report", "/tmp/near.json"]
    )

    assert args.near_contact_fixture_report == "/tmp/near.json"


def test_near_contact_fixture_can_be_raised_and_opened(tmp_path) -> None:
    velocity_report = tmp_path / "velocity.json"
    velocity_report.write_text(
        __import__("json").dumps(
            {
                "report": {
                    "order8_natural_contact_contact_closure_velocity_targets_radps": {
                        "module_0:yaw_dock_mech_joint1": 0.02,
                        "module_1:yaw_dock_mech_joint2": -0.01,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    velocities = _load_fixed_closure_velocity_targets(velocity_report)
    state = {
        "base_pose": [0.5, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0],
        "object_pose": [1.3, 0.0, 0.075, 0.0, 0.0, 0.0, 1.0],
        "joint_positions_rad": {
            "module_0:yaw_dock_mech_joint1": 0.10,
            "module_1:yaw_dock_mech_joint2": -0.20,
        },
    }

    transformed = _transform_near_contact_fixture(
        state,
        height_offset_m=0.15,
        opening_velocity_targets_radps=velocities,
        opening_duration_s=3.0,
    )

    assert transformed["base_pose"][2] == 0.30
    assert transformed["object_pose"][2] == pytest.approx(0.225)
    assert transformed["joint_positions_rad"] == pytest.approx({
        "module_0:yaw_dock_mech_joint1": 0.04,
        "module_1:yaw_dock_mech_joint2": -0.17,
    })


def test_parser_exposes_safe_fixture_trace_options() -> None:
    args = build_parser().parse_args(
        [
            "--near-contact-fixture-report",
            "/tmp/near.json",
            "--fixture-height-offset-m",
            "0.15",
            "--fixture-opening-source-report",
            "/tmp/closure.json",
            "--fixture-opening-duration-s",
            "3.0",
            "--state-trace-path",
            "/tmp/trace.json",
            "--state-trace-frame-stride",
            "2",
        ]
    )

    assert args.fixture_height_offset_m == 0.15
    assert args.fixture_opening_source_report == "/tmp/closure.json"
    assert args.fixture_opening_duration_s == 3.0
    assert args.state_trace_path == "/tmp/trace.json"
    assert args.state_trace_frame_stride == 2


def test_parser_exposes_kinematic_base_isolation() -> None:
    args = build_parser().parse_args(["--kinematic-base-isolation"])
    assert args.kinematic_base_isolation is True


def test_near_contact_fixture_requires_finite_collision_free_complete_state(
    tmp_path,
) -> None:
    report_path = tmp_path / "near.json"
    report_path.write_text(
        __import__("json").dumps(
            {
                "report": {
                    "order8_natural_contact_last_measured_base_module_pose": [
                        0.5,
                        0.0,
                        0.2,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "order8_natural_contact_last_joint_positions_rad": {
                        "module_0:yaw_dock_mech_joint1": 0.1,
                    },
                    "order8_natural_contact_last_measured_object_pose": [
                        1.3,
                        0.0,
                        0.075,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "order8_natural_contact_last_contact_mesh_surface_clearance_m_by_anchor": {
                        "0": 0.002,
                        "1": 0.001,
                    },
                    "order8_natural_contact_qclose_checkpoint_base_pose": None,
                }
            }
        ),
        encoding="utf-8",
    )

    state = _load_near_contact_fixture_report(report_path)

    assert state["base_pose"][0] == 0.5
    assert state["joint_positions_rad"] == {
        "module_0:yaw_dock_mech_joint1": 0.1
    }
    assert state["source_surface_clearance_m_by_anchor"] == {
        "0": 0.002,
        "1": 0.001,
    }
    assert state["source_state_method"] == (
        "terminal_collision_free_near_contact_state_v1"
    )


def test_near_contact_fixture_reuses_recorded_initial_state_from_later_run(
    tmp_path,
) -> None:
    report_path = tmp_path / "completed_diagnostic.json"
    report_path.write_text(
        __import__("json").dumps(
            {
                "report": {
                    "order8_natural_contact_diagnostic_near_contact_base_pose": [
                        0.5,
                        0.0,
                        0.2,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "order8_natural_contact_diagnostic_near_contact_joint_positions_rad": {
                        "module_0:yaw_dock_mech_joint1": 0.1,
                    },
                    "order8_natural_contact_diagnostic_near_contact_object_pose": [
                        1.3,
                        0.0,
                        0.225,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "order8_natural_contact_diagnostic_near_contact_initial_surface_clearance_m_by_anchor": {
                        "0": 0.03,
                        "1": 0.02,
                    },
                    # A later q_close in the same run must not invalidate the
                    # explicitly recorded collision-free initial fixture.
                    "order8_natural_contact_qclose_checkpoint_base_pose": [
                        0.5,
                        0.0,
                        0.2,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    state = _load_near_contact_fixture_report(report_path)

    assert state["base_pose"][2] == pytest.approx(0.2)
    assert state["object_pose"][2] == pytest.approx(0.225)
    assert state["source_surface_clearance_m_by_anchor"] == {
        "0": 0.03,
        "1": 0.02,
    }
    assert state["source_state_method"] == (
        "recorded_diagnostic_initial_near_contact_fixture_v1"
    )


def test_near_contact_fixture_rejects_contacting_state(tmp_path) -> None:
    report_path = tmp_path / "contacting.json"
    report_path.write_text(
        __import__("json").dumps(
            {
                "order8_natural_contact_last_measured_base_module_pose": [
                    0.5,
                    0.0,
                    0.2,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "order8_natural_contact_last_joint_positions_rad": {
                    "module_0:yaw_dock_mech_joint1": 0.1,
                },
                "order8_natural_contact_last_measured_object_pose": [
                    1.3,
                    0.0,
                    0.075,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "order8_natural_contact_last_contact_mesh_surface_clearance_m_by_anchor": {
                    "0": 0.0,
                    "1": 0.001,
                },
                "order8_natural_contact_qclose_checkpoint_base_pose": None,
            }
        ),
        encoding="utf-8",
    )

    try:
        _load_near_contact_fixture_report(report_path)
    except ValueError as exc:
        assert "collision-free" in str(exc)
    else:
        raise AssertionError("contacting near-contact fixture was accepted")


def test_parser_exposes_diagnostic_dock_velocity_limit() -> None:
    args = build_parser().parse_args(["--dock-velocity-limit", "0.5"])

    assert args.dock_velocity_limit == 0.5


def test_parser_exposes_static_qclose_force_isolation() -> None:
    args = build_parser().parse_args(
        [
            "--qclose-fixture-report",
            "/tmp/qclose.json",
            "--zero-qclose-velocities",
        ]
    )

    assert args.qclose_fixture_report == "/tmp/qclose.json"
    assert args.zero_qclose_velocities is True


def test_parser_exposes_diagnostic_dock_armature() -> None:
    args = build_parser().parse_args(["--dock-armature-kg-m2", "0.01"])

    assert args.dock_armature_kg_m2 == 0.01


def test_parser_exposes_diagnostic_peak_torque_window() -> None:
    args = build_parser().parse_args(["--peak-torque-window-s", "1.5"])

    assert args.peak_torque_window_s == 1.5


def test_fixture_can_override_force_and_friction_without_changing_source() -> None:
    source = _Config()

    config = _fast_config(
        source,
        speed_scale=2.0,
        force_ramp_s=4.0,
        object_width_padding_m=0.0,
        full_sequence=False,
        object_friction=0.8,
        contact_dwell_s=0.25,
        selected_gripper_friction=3.0,
        normal_force_target_n=8.0,
    )

    assert config.normal_force_target_per_contact_n == 8.0
    assert config.object_friction == 0.8
    assert config.selected_gripper_friction == 3.0
    assert source.normal_force_target_per_contact_n == 11.0
    assert source.object_friction == 0.6
    assert source.selected_gripper_friction == 2.0

    args = build_parser().parse_args(
        [
            "--normal-force-target-n",
            "8.0",
            "--object-friction",
            "0.8",
            "--selected-gripper-friction",
            "3.0",
        ]
    )
    assert args.normal_force_target_n == 8.0
    assert args.object_friction == 0.8
    assert args.selected_gripper_friction == 3.0


def test_fixture_can_relax_nonprivileged_stall_gate_without_raw_slip_change() -> (
    None
):
    source = _Config()

    config = _fast_config(
        source,
        speed_scale=2.0,
        force_ramp_s=4.0,
        object_width_padding_m=0.0,
        full_sequence=False,
        object_friction=None,
        contact_dwell_s=0.25,
        contact_stall_speed_threshold_mps=0.02,
    )

    assert config.contact_stall_anchor_speed_threshold_mps == 0.02
    assert source.contact_stall_anchor_speed_threshold_mps == 0.0015
    args = build_parser().parse_args(
        ["--contact-stall-speed-threshold-mps", "0.02"]
    )
    assert args.contact_stall_speed_threshold_mps == 0.02


def test_fixture_can_override_slip_telemetry_and_contact_point_limit() -> None:
    source = _Config()

    config = _fast_config(
        source,
        speed_scale=2.0,
        force_ramp_s=4.0,
        object_width_padding_m=0.0,
        full_sequence=False,
        object_friction=None,
        contact_dwell_s=0.25,
        max_slip_speed_mps=0.05,
        max_contact_point_slip_displacement_m=0.05,
    )

    assert config.max_tangential_slip_speed_mps == 0.05
    assert config.max_contact_point_slip_displacement_m == 0.05
    assert source.max_tangential_slip_speed_mps == 0.02
    assert source.max_contact_point_slip_displacement_m == 0.03
    args = build_parser().parse_args(
        [
            "--max-slip-speed-mps",
            "0.05",
            "--max-cumulative-slip-m",
            "0.05",
        ]
    )
    assert args.max_slip_speed_mps == 0.05
    assert args.max_contact_point_slip_displacement_m == 0.05
