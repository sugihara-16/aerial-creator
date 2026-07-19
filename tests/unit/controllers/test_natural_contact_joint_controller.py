from __future__ import annotations

import math

import pytest

from amsrr.controllers.natural_contact_joint_controller import (
    AnchorTaskLinearization,
    DockJointLimit,
    DockJointVector,
    NaturalContactJointController,
    NaturalContactJointControllerConfig,
    position_drive_peak_effort_lead_rad,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL


def _limits(
    count: int,
    *,
    lower: float = -1.0,
    upper: float = 1.0,
    velocity: float = 10.0,
    torque: float = 4.1,
) -> tuple[DockJointLimit, ...]:
    return tuple(
        DockJointLimit(
            position_lower_rad=lower,
            position_upper_rad=upper,
            max_velocity_radps=velocity,
            max_torque_nm=torque,
        )
        for _ in range(count)
    )


def _joint_vector(
    *,
    positions: tuple[float, ...] = (0.0, 0.3, -0.4),
    neutral: tuple[float, ...] | None = None,
    limits: tuple[DockJointLimit, ...] | None = None,
) -> DockJointVector:
    joint_ids = (
        "module_0:pitch_dock_mech_joint1",
        "module_0:yaw_dock_mech_joint2",
        "module_1:pitch_dock_mech_joint2",
    )[: len(positions)]
    return DockJointVector(
        joint_ids=joint_ids,
        positions_rad=positions,
        velocities_radps=tuple(0.0 for _ in positions),
        neutral_positions_rad=neutral or tuple(0.0 for _ in positions),
        limits=limits or _limits(len(positions)),
    )


def _task(
    anchor_id: int,
    *,
    joint_row: tuple[float, ...],
    error_x: float,
    wrench_x: float = 0.0,
) -> AnchorTaskLinearization:
    zeros = tuple(0.0 for _ in joint_row)
    return AnchorTaskLinearization(
        anchor_id=anchor_id,
        task_error=(error_x, 0.0, 0.0, 0.0, 0.0, 0.0),
        jacobian=(joint_row, zeros, zeros, zeros, zeros, zeros),
        wrench_bias=(wrench_x, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _controller(**overrides: float) -> NaturalContactJointController:
    values: dict[str, float | int] = {
        "control_dt_s": 0.1,
        "task_error_gain_per_s": 1.0,
        "dls_damping": 1e-6,
        "neutral_posture_gain_per_s": 0.25,
        "nullspace_velocity_damping": 0.0,
        "max_position_command_lead_rad": 1.0,
        "reachability_absolute_tolerance": 1e-5,
        "reachability_relative_tolerance": 0.01,
        "minimum_simultaneous_anchor_count": 2,
    }
    values.update(overrides)
    return NaturalContactJointController(NaturalContactJointControllerConfig(**values))


def test_whole_structure_ik_keeps_every_joint_and_neutral_regularizes_irrelevant_joint() -> None:
    state = _joint_vector()
    tasks = (
        _task(10, joint_row=(1.0, 0.0, 0.0), error_x=0.1, wrench_x=2.0),
        _task(11, joint_row=(0.0, 1.0, 0.0), error_x=-0.1, wrench_x=-3.0),
    )

    result = _controller().compute(state, tasks)
    command = result.policy_command

    assert tuple(command.joint_position_targets) == state.joint_ids
    assert tuple(command.joint_velocity_targets) == state.joint_ids
    assert tuple(command.joint_torque_bias) == state.joint_ids
    assert command.joint_velocity_targets[state.joint_ids[0]] == pytest.approx(0.1)
    assert command.joint_velocity_targets[state.joint_ids[1]] == pytest.approx(-0.1)
    assert command.joint_velocity_targets[state.joint_ids[2]] == pytest.approx(0.1)
    assert command.joint_position_targets[state.joint_ids[2]] > state.positions_rad[2]
    assert command.joint_torque_bias == {
        state.joint_ids[0]: pytest.approx(2.0),
        state.joint_ids[1]: pytest.approx(-3.0),
        state.joint_ids[2]: pytest.approx(0.0),
    }
    assert result.torque_mapping.unclipped_joint_torque_bias == pytest.approx(
        command.joint_torque_bias
    )
    assert result.torque_mapping.joint_torque_bias == pytest.approx(
        command.joint_torque_bias
    )
    assert result.diagnostics.structural_joint_ids == state.joint_ids
    assert result.diagnostics.structural_variable_count == 3
    assert result.diagnostics.jacobian_column_count == 3
    assert result.diagnostics.neutral_regularized_joint_ids == (state.joint_ids[2],)
    assert result.diagnostics.debug_mask_applied is False
    assert result.reachability.passed is True
    assert result.reachability.status == "reachable"


def test_neutral_regularization_cannot_reverse_primary_task_with_dls() -> None:
    state = _joint_vector(
        positions=(0.5, 0.5),
        neutral=(0.0, 0.0),
    )
    tasks = (
        _task(0, joint_row=(1.0, 1.0), error_x=1.0e-4),
        _task(1, joint_row=(1.0, 1.0), error_x=1.0e-4),
    )

    result = _controller(dls_damping=0.05).compute(state, tasks)
    joint_rates = tuple(result.policy_command.joint_velocity_targets.values())

    # A damped inverse used as the secondary projector leaks the much larger
    # neutral-posture command into J and reverses this small positive task.
    # The true null-space projector must preserve the primary direction.
    assert sum(joint_rates) > 0.0
    assert sum(joint_rates) == pytest.approx(1.0e-4, rel=1.0e-3)


def test_position_drive_lead_uses_peak_effort_without_changing_motor_limit() -> None:
    assert position_drive_peak_effort_lead_rad(
        stiffness_nm_per_rad=200.0,
        peak_effort_nm=4.1,
    ) == pytest.approx(0.0205)


@pytest.mark.parametrize(
    ("stiffness", "peak"),
    [(0.0, 4.1), (math.inf, 4.1), (200.0, 0.0), (200.0, math.nan)],
)
def test_position_drive_lead_rejects_invalid_limits(
    stiffness: float,
    peak: float,
) -> None:
    with pytest.raises(SchemaValidationError, match="position-drive"):
        position_drive_peak_effort_lead_rad(
            stiffness_nm_per_rad=stiffness,
            peak_effort_nm=peak,
        )


def test_debug_mask_is_reported_non_structural_neutral_hold() -> None:
    state = _joint_vector(positions=(0.3, -0.2), neutral=(0.0, 0.1))
    tasks = (
        _task(0, joint_row=(1.0, 0.0), error_x=0.2, wrench_x=3.0),
        _task(1, joint_row=(0.0, 1.0), error_x=0.1, wrench_x=2.0),
    )
    masked_joint = state.joint_ids[0]

    result = _controller().compute(
        state,
        tasks,
        debug_command_mask={masked_joint},
    )

    assert result.policy_command.joint_position_targets[masked_joint] == pytest.approx(0.0)
    assert result.policy_command.joint_velocity_targets[masked_joint] == pytest.approx(0.0)
    assert result.policy_command.joint_torque_bias[masked_joint] == pytest.approx(0.0)
    assert result.torque_mapping.unclipped_joint_torque_bias[masked_joint] == pytest.approx(
        0.0
    )
    assert set(result.policy_command.joint_position_targets) == set(state.joint_ids)
    assert result.diagnostics.structural_joint_ids == state.joint_ids
    assert result.diagnostics.debug_masked_joint_ids == (masked_joint,)
    assert result.diagnostics.debug_mask_applied is True
    assert result.diagnostics.debug_mask_is_non_structural is True


def test_joint_rate_position_and_jacobian_transpose_torque_are_bounded() -> None:
    state = _joint_vector(
        positions=(0.49, 0.0),
        neutral=(0.0, 0.0),
        limits=_limits(2, lower=-0.5, upper=0.5, velocity=0.2, torque=1.0),
    )
    tasks = (
        _task(0, joint_row=(1.0, 0.0), error_x=10.0, wrench_x=10.0),
        _task(1, joint_row=(0.0, 1.0), error_x=0.0, wrench_x=0.0),
    )

    result = _controller().compute(state, tasks)
    first = state.joint_ids[0]

    # The position bound is tighter than the supplied 0.2 rad/s velocity cap.
    assert result.policy_command.joint_velocity_targets[first] == pytest.approx(0.1)
    assert result.policy_command.joint_position_targets[first] == pytest.approx(0.5)
    assert result.policy_command.joint_torque_bias[first] == pytest.approx(1.0)
    assert result.torque_mapping.unclipped_joint_torque_bias[first] == pytest.approx(
        10.0
    )
    assert result.torque_mapping.joint_torque_bias[first] == pytest.approx(1.0)
    assert first in result.diagnostics.velocity_clipped_joint_ids
    assert first in result.diagnostics.position_limited_joint_ids
    assert first in result.diagnostics.torque_clipped_joint_ids


def test_absolute_position_reference_integrates_with_measured_lead_bound() -> None:
    state = _joint_vector(
        positions=(0.0, 0.0),
        neutral=(0.0, 0.0),
    )
    tasks = (
        _task(0, joint_row=(1.0, 0.0), error_x=1.0),
        _task(1, joint_row=(0.0, 1.0), error_x=1.0),
    )
    controller = _controller(max_position_command_lead_rad=0.02)

    first = controller.compute(state, tasks)
    assert tuple(first.policy_command.joint_position_targets.values()) == pytest.approx(
        (0.02, 0.02)
    )
    assert set(
        first.diagnostics.position_command_lead_limited_joint_ids
    ) == set(state.joint_ids)

    advanced_state = _joint_vector(
        positions=(0.01, 0.01),
        neutral=(0.0, 0.0),
    )
    second = controller.compute(
        advanced_state,
        tasks,
        position_reference_rad=first.policy_command.joint_position_targets,
    )
    assert tuple(second.policy_command.joint_position_targets.values()) == pytest.approx(
        (0.03, 0.03)
    )


def test_anchor_hold_outer_loop_preserves_nullspace_position_reference() -> None:
    state = _joint_vector(
        positions=(0.02, -0.02, 0.4),
        neutral=(0.0, 0.0, 0.0),
    )
    tasks = (
        _task(0, joint_row=(1.0, 0.0, 0.0), error_x=0.01),
        _task(1, joint_row=(0.0, 1.0, 0.0), error_x=-0.01),
    )
    reference = {
        state.joint_ids[0]: 0.02,
        state.joint_ids[1]: -0.02,
        state.joint_ids[2]: 0.4,
    }

    result = _controller(
        neutral_posture_gain_per_s=0.0,
        nullspace_velocity_damping=0.0,
    ).compute(
        state,
        tasks,
        position_reference_rad=reference,
    )

    assert result.policy_command.joint_position_targets[state.joint_ids[0]] > 0.02
    assert result.policy_command.joint_position_targets[state.joint_ids[1]] < -0.02
    assert result.policy_command.joint_position_targets[state.joint_ids[2]] == pytest.approx(
        0.4
    )
    assert result.policy_command.joint_velocity_targets[state.joint_ids[2]] == pytest.approx(
        0.0
    )


def test_measured_lead_correction_cannot_raise_velocity_above_motor_limit() -> None:
    state = _joint_vector(
        positions=(0.05, 0.05),
        neutral=(0.0, 0.0),
        limits=_limits(2, velocity=0.1),
    )
    tasks = (
        _task(0, joint_row=(1.0, 0.0), error_x=1.0),
        _task(1, joint_row=(0.0, 1.0), error_x=1.0),
    )

    result = _controller(max_position_command_lead_rad=0.02).compute(
        state,
        tasks,
        position_reference_rad={
            joint_id: 0.0 for joint_id in state.joint_ids
        },
    )

    # Contact moved the measured joints ahead of the stale absolute
    # reference.  The position target must be brought back into the safe
    # position-drive envelope without turning that correction into a 0.3
    # rad/s velocity request.
    assert tuple(result.policy_command.joint_position_targets.values()) == pytest.approx(
        (0.03, 0.03)
    )
    assert tuple(result.policy_command.joint_velocity_targets.values()) == pytest.approx(
        (0.1, 0.1)
    )
    assert set(
        result.diagnostics.position_command_lead_limited_joint_ids
    ) == set(state.joint_ids)


def test_absolute_position_reference_requires_exact_finite_joint_coverage() -> None:
    state = _joint_vector(positions=(0.0, 0.0), neutral=(0.0, 0.0))
    tasks = (
        _task(0, joint_row=(1.0, 0.0), error_x=0.1),
        _task(1, joint_row=(0.0, 1.0), error_x=0.1),
    )
    with pytest.raises(SchemaValidationError, match="cover exactly"):
        _controller().compute(
            state,
            tasks,
            position_reference_rad={state.joint_ids[0]: 0.0},
        )


def test_simultaneous_multi_anchor_reachability_passes_and_conflicting_stack_fails() -> None:
    state = _joint_vector(positions=(0.0, 0.0), neutral=(0.0, 0.0))
    controller = _controller()
    reachable = controller.compute(
        state,
        (
            _task(0, joint_row=(1.0, 0.0), error_x=0.2),
            _task(1, joint_row=(0.0, 1.0), error_x=-0.15),
        ),
    ).reachability
    conflicting = controller.compute(
        state,
        (
            _task(0, joint_row=(1.0, 0.0), error_x=1.0),
            _task(1, joint_row=(1.0, 0.0), error_x=-1.0),
        ),
    ).reachability

    assert reachable.passed is True
    assert reachable.residual_norm <= reachable.tolerance
    assert conflicting.passed is False
    assert conflicting.status == "unreachable_residual"
    assert conflicting.residual_norm > conflicting.tolerance
    assert set(conflicting.per_anchor_residual_norm) == {0, 1}


def test_policy_command_is_centroidal_v2_without_contact_wrench_leakage() -> None:
    state = _joint_vector(positions=(0.0, 0.0), neutral=(0.0, 0.0))
    pose = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    residual = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    result = _controller().compute(
        state,
        (
            _task(0, joint_row=(1.0, 0.0), error_x=0.1, wrench_x=2.0),
            _task(1, joint_row=(0.0, 1.0), error_x=0.1, wrench_x=2.0),
        ),
        desired_body_pose=pose,
        residual_wrench_body=residual,
    )
    command = result.policy_command

    assert command.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert command.desired_body_pose == pose
    assert command.residual_wrench_body == residual
    assert command.contact_tracking_bias == {}
    assert command.desired_anchor_pose_offsets == {}
    assert command.joint_position_bias == {}
    assert command.joint_velocity_bias == {}
    assert not hasattr(command, "internal_wrench_bias")
    command.validate()


def test_vectoring_joint_is_rejected_from_dock_structural_vector() -> None:
    state = DockJointVector(
        joint_ids=("module_0:gimbal1",),
        positions_rad=(0.0,),
        velocities_radps=(0.0,),
        neutral_positions_rad=(0.0,),
        limits=_limits(1),
    )

    with pytest.raises(SchemaValidationError, match="Vectoring joint"):
        _controller().compute(
            state,
            (
                _task(0, joint_row=(1.0,), error_x=0.1),
                _task(1, joint_row=(1.0,), error_x=0.1),
            ),
        )
