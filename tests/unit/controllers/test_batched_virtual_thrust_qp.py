from __future__ import annotations

import math

import pytest
import torch

import amsrr.controllers.batched_virtual_thrust_qp as batched_qp_module
from amsrr.controllers.batched_virtual_thrust_qp import (
    BatchedVirtualThrustQPConfig,
    _project_virtual_channels,
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


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_native_cpu_admm_preserves_applied_commands_and_diagnostics(dtype, monkeypatch):
    from amsrr.feasibility.order9_native_loader import load_order9_posture_native, Order9NativePostureIKUnavailable
    try:
        native = load_order9_posture_native()
    except Order9NativePostureIKUnavailable as error:
        pytest.skip(str(error))
    torch.manual_seed(71)
    batch, rotors = 4, 14
    zero = torch.zeros(batch, rotors, dtype=dtype)
    inputs = dict(desired_wrench_body=torch.randn(batch, 6, dtype=dtype) * 5,
        virtual_x_wrench_columns=torch.randn(batch, rotors, 6, dtype=dtype),
        virtual_z_wrench_columns=torch.randn(batch, rotors, 6, dtype=dtype),
        current_vectoring_angles_rad=zero, previous_rotor_thrusts_n=zero+1,
        previous_vectoring_targets_rad=zero, thrust_min_n=zero, thrust_max_n=zero+10,
        vectoring_lower_rad=zero-1, vectoring_upper_rad=zero+1,
        vectoring_velocity_limit_radps=zero+2, control_dt_s=.02,
        unsupported_wrench_tolerance=2., rotor_mask=torch.ones(batch, rotors, dtype=torch.bool))
    inputs['rotor_mask'][0] = False
    inputs['rotor_mask'][1, ::2] = False
    inputs['vectoring_lower_rad'][2] = 0
    inputs['vectoring_upper_rad'][2] = 0
    with torch.no_grad():
        monkeypatch.setattr(batched_qp_module, '_native_cpu_admm', False)
        reference = solve_batched_virtual_thrust_qp(**inputs)
        monkeypatch.setattr(batched_qp_module, '_native_cpu_admm', native)
        actual = solve_batched_virtual_thrust_qp(**inputs)
    for name in reference.__dataclass_fields__:
        a, b = getattr(reference, name), getattr(actual, name)
        if a.dtype == torch.bool:
            assert torch.equal(a, b), name
        else:
            tolerance = 2e-4 if dtype == torch.float32 else 2e-11
            torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)


@pytest.fixture
def cusolver_backend():
    previous = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library("cusolver")
    try:
        yield
    finally:
        torch.backends.cuda.preferred_linalg_library(previous)
        batched_qp_module._admm_cuda_cache = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph needs CUDA')
@pytest.mark.parametrize('batch', [1, 3])
def test_cuda_admm_replay_uses_new_inputs_and_owns_outputs(monkeypatch, cusolver_backend, batch):
    # Compare the same production solve with Python launch vs captured launches.
    desired = torch.tensor([[2., 0., 5., 0., 0., 0.]], device='cuda').repeat(batch, 1)
    x, z = (v.float().cuda() for v in _columns(batch))
    zero = torch.zeros(batch, 1, device='cuda')
    inputs = dict(desired_wrench_body=desired, virtual_x_wrench_columns=x,
        virtual_z_wrench_columns=z, current_vectoring_angles_rad=zero,
        previous_rotor_thrusts_n=zero, previous_vectoring_targets_rad=zero,
        thrust_min_n=zero, thrust_max_n=zero+10, vectoring_lower_rad=zero-1,
        vectoring_upper_rad=zero+1, vectoring_velocity_limit_radps=zero+100,
        control_dt_s=.02, unsupported_wrench_tolerance=2.)
    captured = batched_qp_module._admm_cuda
    with torch.no_grad():
        first = solve_batched_virtual_thrust_qp(**inputs)
        saved = first.rotor_thrusts_n.clone()
        inputs['desired_wrench_body'] = desired * .8
        second = solve_batched_virtual_thrust_qp(**inputs)
        torch.testing.assert_close(first.rotor_thrusts_n, saved, atol=0, rtol=0)
        monkeypatch.setattr(batched_qp_module, '_admm_cuda', batched_qp_module._admm_iterations)
        reference = solve_batched_virtual_thrust_qp(**inputs)
        for name in reference.__dataclass_fields__:
            torch.testing.assert_close(getattr(second, name), getattr(reference, name), atol=0, rtol=0)
        monkeypatch.setattr(batched_qp_module, '_admm_cuda', captured)


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


def test_batched_qp_physical_feasibility_does_not_require_admm_optimality() -> None:
    """A projected command inside the wrench tolerance is physically feasible."""

    desired = torch.tensor(
        [[0.1, 0.0, 0.1, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    x_column, z_column = _columns()
    scalar = torch.zeros((1, 1), dtype=desired.dtype)
    result = solve_batched_virtual_thrust_qp(
        desired_wrench_body=desired,
        virtual_x_wrench_columns=x_column,
        virtual_z_wrench_columns=z_column,
        current_vectoring_angles_rad=scalar,
        previous_rotor_thrusts_n=scalar,
        previous_vectoring_targets_rad=scalar,
        thrust_min_n=scalar,
        thrust_max_n=torch.full_like(scalar, 10.0),
        vectoring_lower_rad=torch.full_like(scalar, -1.0),
        vectoring_upper_rad=torch.full_like(scalar, 1.0),
        vectoring_velocity_limit_radps=torch.full_like(scalar, 100.0),
        control_dt_s=0.1,
        unsupported_wrench_tolerance=1.0,
        config=BatchedVirtualThrustQPConfig(
            regularization_weight=0.0,
            previous_command_weight=0.0,
            max_iterations=1,
            projection_iterations=1,
            absolute_tolerance=1.0e-12,
            relative_tolerance=1.0e-12,
        ),
    )

    assert result.solver_converged.tolist() == [False]
    assert result.residual_norm.item() < 1.0
    assert result.feasible.tolist() == [True]


def test_batched_qp_runs_configured_admm_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not treat small ADMM iterate deltas as wrench-objective convergence."""

    projection_calls = 0
    original_projection = batched_qp_module._project_virtual_channels

    def counting_projection(*args, **kwargs):
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(*args, **kwargs)

    monkeypatch.setattr(
        batched_qp_module, "_project_virtual_channels", counting_projection
    )
    desired = torch.zeros((1, 6), dtype=torch.float64)
    x_column, z_column = _columns()
    scalar = torch.zeros((1, 1), dtype=desired.dtype)
    result = solve_batched_virtual_thrust_qp(
        desired_wrench_body=desired,
        virtual_x_wrench_columns=x_column,
        virtual_z_wrench_columns=z_column,
        current_vectoring_angles_rad=scalar,
        previous_rotor_thrusts_n=scalar,
        previous_vectoring_targets_rad=scalar,
        thrust_min_n=scalar,
        thrust_max_n=torch.full_like(scalar, 10.0),
        vectoring_lower_rad=torch.full_like(scalar, -1.0),
        vectoring_upper_rad=torch.full_like(scalar, 1.0),
        vectoring_velocity_limit_radps=torch.full_like(scalar, 100.0),
        control_dt_s=0.1,
        unsupported_wrench_tolerance=1.0,
        config=BatchedVirtualThrustQPConfig(
            regularization_weight=0.0,
            previous_command_weight=0.0,
            max_iterations=17,
            projection_iterations=1,
        ),
    )

    # One projection initializes z, followed by exactly max_iterations ADMM
    # projections.  The former early-exit implementation stopped at iteration 8
    # for this already-stationary input.
    assert projection_calls == 18
    assert result.solver_converged.tolist() == [True]


def test_exact_virtual_channel_projection_matches_converged_dykstra_reference() -> None:
    generator = torch.Generator().manual_seed(147)
    values = 8.0 * torch.randn((3, 7, 2), generator=generator, dtype=torch.float64)
    minimum_z = torch.rand((3, 7), generator=generator, dtype=torch.float64)
    maximum_z = minimum_z + 2.0 + 5.0 * torch.rand(
        (3, 7), generator=generator, dtype=torch.float64
    )
    maximum_x = maximum_z.clone()
    angle_lower = -1.2 + 0.8 * torch.rand(
        (3, 7), generator=generator, dtype=torch.float64
    )
    angle_upper = 0.4 + 0.8 * torch.rand(
        (3, 7), generator=generator, dtype=torch.float64
    )
    pair_i = torch.tensor(
        (0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4),
        dtype=torch.long,
    )
    pair_j = torch.tensor(
        (1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5),
        dtype=torch.long,
    )

    projected = _project_virtual_channels(
        values,
        minimum_virtual_z=minimum_z,
        maximum_virtual_z=maximum_z,
        maximum_virtual_x=maximum_x,
        angle_lower=angle_lower,
        angle_upper=angle_upper,
        iterations=12,
        pair_i=pair_i,
        pair_j=pair_j,
    )
    reference = _dykstra_projection_reference(
        values,
        minimum_z=minimum_z,
        maximum_z=maximum_z,
        maximum_x=maximum_x,
        angle_lower=angle_lower,
        angle_upper=angle_upper,
        iterations=512,
    )

    torch.testing.assert_close(projected, reference, rtol=0.0, atol=2.0e-9)


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


def _dykstra_projection_reference(
    values: torch.Tensor,
    *,
    minimum_z: torch.Tensor,
    maximum_z: torch.Tensor,
    maximum_x: torch.Tensor,
    angle_lower: torch.Tensor,
    angle_upper: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    tangent_upper = torch.tan(angle_upper)
    tangent_lower = torch.tan(angle_lower)
    value = values.clone()
    box_correction = torch.zeros_like(value)
    upper_correction = torch.zeros_like(value)
    lower_correction = torch.zeros_like(value)
    for _ in range(iterations):
        candidate = value + box_correction
        box = torch.stack(
            (
                candidate[..., 0].clamp(-maximum_x, maximum_x),
                candidate[..., 1].clamp(minimum_z, maximum_z),
            ),
            dim=-1,
        )
        box_correction = candidate - box
        value = box
        candidate = value + upper_correction
        violation = (
            candidate[..., 0] - tangent_upper * candidate[..., 1]
        ).clamp_min(0.0)
        multiplier = violation / (1.0 + tangent_upper.square())
        upper = torch.stack(
            (
                candidate[..., 0] - multiplier,
                candidate[..., 1] + multiplier * tangent_upper,
            ),
            dim=-1,
        )
        upper_correction = candidate - upper
        value = upper
        candidate = value + lower_correction
        violation = (
            -candidate[..., 0] + tangent_lower * candidate[..., 1]
        ).clamp_min(0.0)
        multiplier = violation / (1.0 + tangent_lower.square())
        lower = torch.stack(
            (
                candidate[..., 0] + multiplier,
                candidate[..., 1] - multiplier * tangent_lower,
            ),
            dim=-1,
        )
        lower_correction = candidate - lower
        value = lower
    return value
