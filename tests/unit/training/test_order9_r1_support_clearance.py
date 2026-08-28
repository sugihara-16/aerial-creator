from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_posture_collision_object,
)
from amsrr.training.order9_r1_fast_screen import (
    ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL,
    ORDER9_R1_FAST_SCREEN_VERSION,
    Order9R1FastScreenResult,
)
from amsrr.training.order9_r1_execution_clearance import (
    ORDER9_R1_EXECUTION_ROBUST_CLEARANCE_M,
    ORDER9_R1_MEASURED_MINIMUM_REQUIRED_CLEARANCE_M,
    build_order9_r1_execution_clearance_contract,
    require_order9_r1_v3_execution_clearance,
)
from amsrr.training.order9_r1_randomization import apply_order9_r1_pose_offsets
from amsrr.training.order9_r1_pi_l_action_diagnostic import (
    mask_order9_r1_joint_and_compression_inputs,
)
from amsrr.training.order9_r1_pi_l_isolated_action_diagnostic import (
    mask_order9_r1_isolated_action_inputs,
    protected_rollout_source_without_process_exit,
)
from amsrr.training.order9_r1_support_clearance import (
    ORDER9_R1_ROBUST_SUPPORT_CLEARANCE_M,
    build_order9_r1_frozen_support_collision_object,
    require_order9_r1_v2_support_clearance,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec


def test_r1_v2_freezes_source_support_while_object_moves() -> None:
    source = _task()
    randomized = apply_order9_r1_pose_offsets(
        source,
        seed=71,
        sample_index=0,
        x_offset_m=-0.01,
        y_offset_m=-0.01,
        yaw_offset_rad=-0.05,
        authorization_kind="approved_calibration_protocol",
        authorization_path="configs/r1.yaml",
        authorization_sha256="a" * 64,
    ).task_spec

    source_collision = build_order9_c3_posture_collision_object(source)
    moving_collision = build_order9_c3_posture_collision_object(randomized)
    frozen_collision, contract = (
        build_order9_r1_frozen_support_collision_object(
            source_task_spec=source,
            randomized_task_spec=randomized,
        )
    )

    assert (
        moving_collision.environment_boxes
        != source_collision.environment_boxes
    )
    assert (
        frozen_collision.environment_boxes
        == source_collision.environment_boxes
    )
    assert (
        frozen_collision.initial_pose_world
        == randomized.scene.objects[0].pose_world
    )
    assert (
        contract.requested_clearance_m == ORDER9_R1_ROBUST_SUPPORT_CLEARANCE_M
    )
    assert contract.minimum_admissible_clearance_m == pytest.approx(0.0118)


def test_r1_v2_clearance_gate_rejects_old_five_millimetre_screen() -> None:
    source = _task()
    _collision, contract = build_order9_r1_frozen_support_collision_object(
        source_task_spec=source,
        randomized_task_spec=source,
    )
    accepted = _screen(clearance_m=0.0118)
    require_order9_r1_v2_support_clearance(screen=accepted, contract=contract)

    with pytest.raises(SchemaValidationError, match="execution clearance"):
        require_order9_r1_v2_support_clearance(
            screen=replace(accepted, minimum_collision_clearance_m=0.005),
            contract=contract,
        )


def test_r1_v3_clearance_includes_tracking_and_pi_l_consumption() -> None:
    source = _task()
    _collision, contract = build_order9_r1_execution_clearance_contract(
        source_task_spec=source,
        randomized_task_spec=source,
    )

    assert ORDER9_R1_MEASURED_MINIMUM_REQUIRED_CLEARANCE_M == pytest.approx(
        0.02030075881572687
    )
    assert (
        contract.requested_clearance_m
        == ORDER9_R1_EXECUTION_ROBUST_CLEARANCE_M
    )
    assert contract.requested_clearance_m == 0.022
    require_order9_r1_v3_execution_clearance(
        screen=_screen(clearance_m=0.0218), contract=contract
    )
    with pytest.raises(SchemaValidationError, match="execution-error"):
        require_order9_r1_v3_execution_clearance(
            screen=_screen(clearance_m=0.0203), contract=contract
        )


def test_r1_pi_l_diagnostic_keeps_only_joint_and_normal_compression() -> None:
    global_action = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    joint_action = torch.arange(48, dtype=torch.float32).reshape(2, 2, 12)
    contact_action = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6)

    masked_global, masked_joint, masked_contact = (
        mask_order9_r1_joint_and_compression_inputs(
            normalized_global_action=global_action,
            normalized_joint_action=joint_action,
            normalized_contact_action=contact_action,
        )
    )

    assert torch.equal(masked_global, torch.zeros_like(global_action))
    assert torch.equal(masked_joint, joint_action)
    assert torch.equal(masked_contact[:, :, 0], contact_action[:, :, 0])
    assert torch.equal(
        masked_contact[:, :, 1:], torch.zeros_like(contact_action[:, :, 1:])
    )


@pytest.mark.parametrize("mode", ["joint_only", "contact_normal_only"])
def test_r1_pi_l_isolated_diagnostic_keeps_exactly_one_path(mode: str) -> None:
    global_action = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    joint_action = torch.arange(48, dtype=torch.float32).reshape(2, 2, 12)
    contact_action = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6)

    masked_global, masked_joint, masked_contact = (
        mask_order9_r1_isolated_action_inputs(
            mode=mode,  # type: ignore[arg-type]
            normalized_global_action=global_action,
            normalized_joint_action=joint_action,
            normalized_contact_action=contact_action,
        )
    )

    assert torch.equal(masked_global, torch.zeros_like(global_action))
    assert torch.equal(
        masked_joint,
        (
            joint_action
            if mode == "joint_only"
            else torch.zeros_like(joint_action)
        ),
    )
    expected_contact = torch.zeros_like(contact_action)
    if mode == "contact_normal_only":
        expected_contact[:, :, 0] = contact_action[:, :, 0]
    assert torch.equal(masked_contact, expected_contact)


def test_r1_pi_l_isolated_diagnostic_delays_only_protected_finalizer() -> None:
    source = (
        "value = 1\n"
        "finally:\n"
        "    simulation_app.close()\n"
        "raise SystemExit(_exit_code)\n"
    )

    transformed = protected_rollout_source_without_process_exit(source)

    assert transformed == "value = 1\nfinally:\n    pass\n"


def _task():
    return build_order8_grasp_carry_task_spec(
        object_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.20,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
    )


def _screen(*, clearance_m: float) -> Order9R1FastScreenResult:
    return Order9R1FastScreenResult(
        screen_version=ORDER9_R1_FAST_SCREEN_VERSION,
        execution_model=ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL,
        candidate_id="r1-v2-unit",
        candidate_task_hash="a" * 64,
        morphology_hash="b" * 64,
        physical_model_hash="c" * 64,
        resolved_path_hash="d" * 64,
        accepted=True,
        eligible_for_full_control_test=True,
        violation_codes=(),
        checked_phases=("approach", "contact_acquisition"),
        window_count=2,
        resolved_knot_count=4,
        collision_evidence_window_count=2,
        configuration_plan_collision_check_count=4,
        grasp_pose_reached=True,
        final_maintained_contact_count=2,
        minimum_joint_limit_margin_rad=0.2,
        minimum_normalized_joint_limit_reserve=0.1,
        maximum_body_tilt_rad=0.0,
        maximum_joint_rate_rad_s=0.1,
        minimum_joint_rate_margin_rad_s=0.9,
        minimum_collision_clearance_m=clearance_m,
        maximum_collision_violating_pair_count=0,
        screen_wall_time_s=0.001,
    )
