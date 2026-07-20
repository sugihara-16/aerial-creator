from __future__ import annotations

import math

import pytest
import torch

from amsrr.controllers.batched_virtual_thrust_qp import (
    BatchedVirtualThrustQPConfig,
    solve_batched_virtual_thrust_qp,
)
from amsrr.controllers.qp_allocator_interface import (
    QPAllocationProblem,
    VirtualThrustQPAllocator,
)
from amsrr.controllers.rigid_body_model import (
    RigidBodyControlModel,
    RotorControlElement,
)


def _columns(batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    x_column = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=torch.float64
    ).repeat(batch_size, 1, 1)
    z_column = torch.tensor(
        [[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64
    ).repeat(batch_size, 1, 1)
    return x_column, z_column


def _solve(
    desired: torch.Tensor,
    *,
    thrust_max_n: float = 10.0,
    velocity_limit_radps: float = 100.0,
    dt_s: float = 0.1,
    rotor_mask: torch.Tensor | None = None,
):
    batch_size = desired.shape[0]
    x_column, z_column = _columns(batch_size)
    scalar = torch.zeros((batch_size, 1), dtype=desired.dtype)
    return solve_batched_virtual_thrust_qp(
        desired_wrench_body=desired,
        virtual_x_wrench_columns=x_column,
        virtual_z_wrench_columns=z_column,
        current_vectoring_angles_rad=scalar,
        previous_rotor_thrusts_n=scalar,
        previous_vectoring_targets_rad=scalar,
        thrust_min_n=scalar,
        thrust_max_n=torch.full_like(scalar, thrust_max_n),
        vectoring_lower_rad=torch.full_like(scalar, -1.0),
        vectoring_upper_rad=torch.full_like(scalar, 1.0),
        vectoring_velocity_limit_radps=torch.full_like(
            scalar, velocity_limit_radps
        ),
        control_dt_s=dt_s,
        unsupported_wrench_tolerance=100.0,
        rotor_mask=rotor_mask,
        config=BatchedVirtualThrustQPConfig(
            regularization_weight=0.0,
            previous_command_weight=0.0,
            max_iterations=96,
            projection_iterations=16,
            absolute_tolerance=1.0e-7,
            relative_tolerance=1.0e-7,
        ),
    )


def _scalar_model(
    *, thrust_max_n: float, velocity_limit_radps: float
) -> RigidBodyControlModel:
    rotor = RotorControlElement(
        global_rotor_id="module_0:thrust_1",
        module_id=0,
        rotor_id="thrust_1",
        thrust_frame_link="thrust_1",
        origin_body=(0.0, 0.0, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        thrust_min_n=0.0,
        thrust_max_n=thrust_max_n,
        reaction_torque_coeff_nm_per_n=0.0,
        reaction_torque_axis_body=(0.0, 0.0, 1.0),
        vectoring_joint_ids=["module_0:gimbal1"],
        virtual_x_axis_body=(1.0, 0.0, 0.0),
        virtual_z_axis_body=(0.0, 0.0, 1.0),
        allocation_column_body=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )
    return RigidBodyControlModel(
        model_id="unit",
        graph_id="unit",
        base_module_id=0,
        body_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        total_mass_kg=1.0,
        center_of_mass_body=(0.0, 0.0, 0.0),
        inertia_body=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        rotor_elements=[rotor],
        rotor_origins_body={rotor.global_rotor_id: rotor.origin_body},
        rotor_axes_body={rotor.global_rotor_id: rotor.axis_body},
        allocation_matrix_body=[[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]],
        vectoring_joint_axes_body={"module_0:gimbal1": (1.0, 0.0, 0.0)},
        dock_actuator_ids=[],
        active_actuator_limits={
            rotor.global_rotor_id: {
                "lower": 0.0,
                "upper": thrust_max_n,
                "velocity": None,
                "effort": None,
            },
            "module_0:gimbal1": {
                "lower": -1.0,
                "upper": 1.0,
                "velocity": velocity_limit_radps,
                "effort": 1.0,
            },
        },
        current_joint_positions={"module_0:gimbal1": 0.0},
    )


def test_batched_qp_recovers_unconstrained_virtual_channels() -> None:
    desired = torch.tensor(
        [[2.0, 0.0, 5.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )

    result = _solve(desired)

    assert result.solver_converged.tolist() == [True]
    assert result.virtual_channel_solution[0, 0].tolist() == pytest.approx(
        [2.0, 5.0], abs=1.0e-7
    )
    assert result.rotor_thrusts_n.item() == pytest.approx(math.sqrt(29.0))
    assert result.vectoring_joint_targets_rad.item() == pytest.approx(
        math.atan2(2.0, 5.0)
    )
    assert result.residual_norm.item() == pytest.approx(0.0, abs=1.0e-7)


def test_batched_qp_matches_scalar_qp_at_rate_and_thrust_limits() -> None:
    desired = torch.tensor(
        [[10.0, 0.0, 10.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    batched = _solve(
        desired, thrust_max_n=10.0, velocity_limit_radps=0.5, dt_s=0.1
    )
    scalar_allocator = VirtualThrustQPAllocator()
    scalar_allocator.regularization_weight = 0.0
    scalar_allocator.previous_command_weight = 0.0
    scalar = scalar_allocator.allocate(
        QPAllocationProblem(
            desired_wrench_body=desired[0].tolist(),
            rotors=[],
            rigid_body_model=_scalar_model(
                thrust_max_n=10.0, velocity_limit_radps=0.5
            ),
            control_dt_s=0.1,
            unsupported_wrench_tolerance=100.0,
        )
    )

    assert batched.solver_converged.tolist() == [True]
    assert batched.rotor_thrusts_n.item() == pytest.approx(
        scalar.rotor_thrusts_n["module_0:thrust_1"], abs=2.0e-5
    )
    assert batched.vectoring_joint_targets_rad.item() == pytest.approx(
        scalar.vectoring_joint_targets["module_0:gimbal1"], abs=2.0e-5
    )
    assert batched.residual_norm.item() == pytest.approx(
        scalar.residual_norm, abs=2.0e-5
    )


def test_batched_qp_padded_rotor_is_fixed_to_zero() -> None:
    desired = torch.tensor(
        [[2.0, 0.0, 5.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )

    result = _solve(desired, rotor_mask=torch.tensor([[False]]))

    assert result.rotor_thrusts_n.item() == 0.0
    assert result.virtual_channel_solution[0, 0].tolist() == [0.0, 0.0]
    assert result.residual_wrench_body[0].tolist() == desired[0].tolist()


def test_batched_qp_rejects_mismatched_shapes() -> None:
    desired = torch.zeros((2, 6), dtype=torch.float32)
    x_column, z_column = _columns(1)
    scalar = torch.zeros((2, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="batch dimensions"):
        solve_batched_virtual_thrust_qp(
            desired_wrench_body=desired,
            virtual_x_wrench_columns=x_column.float(),
            virtual_z_wrench_columns=z_column.float(),
            current_vectoring_angles_rad=scalar,
            previous_rotor_thrusts_n=scalar,
            previous_vectoring_targets_rad=scalar,
            thrust_min_n=scalar,
            thrust_max_n=scalar + 1.0,
            vectoring_lower_rad=scalar - 1.0,
            vectoring_upper_rad=scalar + 1.0,
            vectoring_velocity_limit_radps=scalar + 1.0,
            control_dt_s=0.02,
            unsupported_wrench_tolerance=1.0,
        )
