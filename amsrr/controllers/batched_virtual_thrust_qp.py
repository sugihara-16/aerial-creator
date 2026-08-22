from __future__ import annotations

"""GPU-batched form of the production virtual-thrust allocation QP.

The scalar :class:`VirtualThrustQPAllocator` uses SciPy/SLSQP and is retained
for the full-fidelity controller and the isolated Order 9 shadow checker.  A
Python solver invocation per Isaac environment is not viable on the training
hot path, so this module solves the *same* convex objective and the same box /
vectoring-angle constraints with batched ADMM.  It is an allocator backend,
not a policy projection: ``pi_L`` still emits ``PolicyCommand`` intent and the
controller remains the sole owner of rotor and vectoring commands.
"""

import math
from dataclasses import dataclass

import torch


_SMALL_BATCH_COMPILE_LIMIT = 8
_compiled_small_batch_projection = None
_small_batch_compilation_disabled = False


@dataclass(frozen=True)
class BatchedVirtualThrustQPConfig:
    regularization_weight: float = 1.0e-8
    previous_command_weight: float = 1.0e-8
    admm_penalty: float = 1.0
    max_iterations: int = 64
    projection_iterations: int = 12
    absolute_tolerance: float = 2.0e-5
    relative_tolerance: float = 2.0e-5
    vectoring_deadband_n: float = 1.0e-7

    def validate(self) -> None:
        for name in (
            "regularization_weight",
            "previous_command_weight",
            "absolute_tolerance",
            "relative_tolerance",
            "vectoring_deadband_n",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.admm_penalty) or self.admm_penalty <= 0.0:
            raise ValueError("admm_penalty must be finite and positive")
        if self.max_iterations < 1 or self.projection_iterations < 1:
            raise ValueError("batched QP iteration counts must be positive")


@dataclass(frozen=True)
class BatchedVirtualThrustQPResult:
    rotor_thrusts_n: torch.Tensor
    vectoring_joint_targets_rad: torch.Tensor
    virtual_channel_solution: torch.Tensor
    achieved_wrench_body: torch.Tensor
    residual_wrench_body: torch.Tensor
    residual_norm: torch.Tensor
    feasible: torch.Tensor
    solver_converged: torch.Tensor
    thrust_clipped: torch.Tensor
    vectoring_clipped: torch.Tensor
    primal_residual_norm: torch.Tensor
    dual_residual_norm: torch.Tensor
    objective: torch.Tensor


def solve_batched_virtual_thrust_qp(
    *,
    desired_wrench_body: torch.Tensor,
    virtual_x_wrench_columns: torch.Tensor,
    virtual_z_wrench_columns: torch.Tensor,
    current_vectoring_angles_rad: torch.Tensor,
    previous_rotor_thrusts_n: torch.Tensor,
    previous_vectoring_targets_rad: torch.Tensor,
    thrust_min_n: torch.Tensor,
    thrust_max_n: torch.Tensor,
    vectoring_lower_rad: torch.Tensor,
    vectoring_upper_rad: torch.Tensor,
    vectoring_velocity_limit_radps: torch.Tensor,
    control_dt_s: float,
    unsupported_wrench_tolerance: float,
    rotor_mask: torch.Tensor | None = None,
    config: BatchedVirtualThrustQPConfig | None = None,
) -> BatchedVirtualThrustQPResult:
    """Solve one virtual-x/z allocation QP for each batch row.

    ``virtual_*_wrench_columns`` have shape ``[batch, rotor, 6]``.  Every
    rotor occupies two interleaved optimization variables ``(fx, fz)``.  A
    false ``rotor_mask`` fixes both variables and both output commands at zero,
    which permits a topology bucket to use a padded rotor dimension.
    """

    resolved = config or BatchedVirtualThrustQPConfig()
    resolved.validate()
    if not math.isfinite(float(control_dt_s)) or control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    if (
        not math.isfinite(float(unsupported_wrench_tolerance))
        or unsupported_wrench_tolerance < 0.0
    ):
        raise ValueError(
            "unsupported_wrench_tolerance must be finite and non-negative"
        )
    _validate_inputs(
        desired_wrench_body=desired_wrench_body,
        virtual_x_wrench_columns=virtual_x_wrench_columns,
        virtual_z_wrench_columns=virtual_z_wrench_columns,
        current_vectoring_angles_rad=current_vectoring_angles_rad,
        previous_rotor_thrusts_n=previous_rotor_thrusts_n,
        previous_vectoring_targets_rad=previous_vectoring_targets_rad,
        thrust_min_n=thrust_min_n,
        thrust_max_n=thrust_max_n,
        vectoring_lower_rad=vectoring_lower_rad,
        vectoring_upper_rad=vectoring_upper_rad,
        vectoring_velocity_limit_radps=vectoring_velocity_limit_radps,
        rotor_mask=rotor_mask,
    )
    batch_size, rotor_count, _ = virtual_x_wrench_columns.shape
    device = desired_wrench_body.device
    dtype = desired_wrench_body.dtype
    mask = (
        torch.ones((batch_size, rotor_count), dtype=torch.bool, device=device)
        if rotor_mask is None
        else rotor_mask.to(device=device, dtype=torch.bool)
    )
    tensors = (
        virtual_x_wrench_columns,
        virtual_z_wrench_columns,
        current_vectoring_angles_rad,
        previous_rotor_thrusts_n,
        previous_vectoring_targets_rad,
        thrust_min_n,
        thrust_max_n,
        vectoring_lower_rad,
        vectoring_upper_rad,
        vectoring_velocity_limit_radps,
    )
    (
        x_columns,
        z_columns,
        current_angle,
        previous_thrust,
        previous_angle,
        thrust_min,
        thrust_max,
        hard_angle_lower,
        hard_angle_upper,
        angle_velocity,
    ) = tuple(value.to(device=device, dtype=dtype) for value in tensors)

    angle_delta = angle_velocity.abs() * float(control_dt_s)
    angle_lower = torch.maximum(hard_angle_lower, current_angle - angle_delta)
    angle_upper = torch.minimum(hard_angle_upper, current_angle + angle_delta)
    invalid_interval = angle_lower > angle_upper
    clamped_current = torch.minimum(
        torch.maximum(current_angle, hard_angle_lower), hard_angle_upper
    )
    angle_lower = torch.where(invalid_interval, clamped_current, angle_lower)
    angle_upper = torch.where(invalid_interval, clamped_current, angle_upper)
    maximum_angle = math.pi / 2.0 - 1.0e-4
    angle_lower = angle_lower.clamp(min=-maximum_angle, max=maximum_angle)
    angle_upper = angle_upper.clamp(min=-maximum_angle, max=maximum_angle)

    max_abs_angle = torch.maximum(angle_lower.abs(), angle_upper.abs())
    minimum_virtual_z = torch.maximum(
        torch.zeros_like(thrust_min), thrust_min * torch.cos(max_abs_angle)
    )
    # Padded rotors are fixed at the origin and contribute no wrench.
    zero = torch.zeros((), device=device, dtype=dtype)
    minimum_virtual_z = torch.where(mask, minimum_virtual_z, zero)
    effective_thrust_max = torch.where(mask, thrust_max, zero)
    effective_x_columns = x_columns * mask.unsqueeze(-1).to(dtype)
    effective_z_columns = z_columns * mask.unsqueeze(-1).to(dtype)

    allocation_matrix = torch.stack(
        (effective_x_columns, effective_z_columns), dim=2
    ).reshape(batch_size, rotor_count * 2, 6).transpose(1, 2)
    previous_virtual = torch.stack(
        (
            previous_thrust * torch.sin(previous_angle),
            previous_thrust * torch.cos(previous_angle),
        ),
        dim=-1,
    )
    previous_virtual = previous_virtual * mask.unsqueeze(-1).to(dtype)
    previous_flat = previous_virtual.reshape(batch_size, rotor_count * 2)

    identity = torch.eye(
        rotor_count * 2, device=device, dtype=dtype
    ).unsqueeze(0)
    hessian = allocation_matrix.transpose(1, 2) @ allocation_matrix
    hessian = hessian + (
        resolved.regularization_weight + resolved.previous_command_weight
    ) * identity
    rhs = (
        allocation_matrix.transpose(1, 2)
        @ desired_wrench_body.unsqueeze(-1)
    ).squeeze(-1)
    rhs = rhs + resolved.previous_command_weight * previous_flat
    factored = torch.linalg.cholesky(
        hessian + resolved.admm_penalty * identity
    )

    projection_kwargs = {
        "minimum_virtual_z": minimum_virtual_z,
        "maximum_virtual_z": effective_thrust_max,
        "maximum_virtual_x": effective_thrust_max,
        "angle_lower": angle_lower,
        "angle_upper": angle_upper,
        "iterations": resolved.projection_iterations,
        "pair_i": torch.tensor(
            (0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4),
            device=device,
            dtype=torch.long,
        ),
        "pair_j": torch.tensor(
            (1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5),
            device=device,
            dtype=torch.long,
        ),
    }
    projection = (
        _project_virtual_channels_compiled_small_batch
        if device.type == "cuda" and batch_size <= _SMALL_BATCH_COMPILE_LIMIT
        else _project_virtual_channels
    )
    z_value = projection(
        previous_virtual, **projection_kwargs
    ).reshape(batch_size, rotor_count * 2)
    x_value = z_value.clone()
    dual = torch.zeros_like(x_value)
    previous_z = z_value
    for _ in range(resolved.max_iterations):
        solve_rhs = rhs + resolved.admm_penalty * (z_value - dual)
        x_value = torch.cholesky_solve(
            solve_rhs.unsqueeze(-1), factored
        ).squeeze(-1)
        previous_z = z_value
        projected = projection(
            (x_value + dual).reshape(batch_size, rotor_count, 2),
            **projection_kwargs,
        )
        z_value = projected.reshape(batch_size, rotor_count * 2)
        dual = dual + x_value - z_value

    # The primal/dual stopping test is intentionally diagnostic-only.  For this
    # ill-conditioned wrench allocator, small iterate deltas do not imply that
    # the wrench objective has reached the physical-feasibility tolerance.  A
    # data-dependent early exit therefore changes controller semantics and has
    # produced false QPID-infeasible terminals in continuous validation.
    primal_norm, dual_norm, solver_converged = _admm_convergence(
        x_value=x_value,
        z_value=z_value,
        previous_z=previous_z,
        penalty=resolved.admm_penalty,
        rotor_count=rotor_count,
        absolute_tolerance=resolved.absolute_tolerance,
        relative_tolerance=resolved.relative_tolerance,
    )

    channels = z_value.reshape(batch_size, rotor_count, 2)
    raw_fx = channels[..., 0]
    raw_fz = channels[..., 1].clamp_min(0.0)
    raw_thrust = torch.sqrt(raw_fx.square() + raw_fz.square())
    raw_target = torch.atan2(raw_fx, raw_fz)
    thrust = torch.minimum(torch.maximum(raw_thrust, thrust_min), thrust_max)
    target = torch.minimum(torch.maximum(raw_target, angle_lower), angle_upper)
    target = torch.where(
        thrust <= resolved.vectoring_deadband_n,
        torch.minimum(torch.maximum(current_angle, angle_lower), angle_upper),
        target,
    )
    thrust_clipped = mask & (
        (raw_thrust < thrust_min - 1.0e-8)
        | (raw_thrust > thrust_max + 1.0e-8)
    )
    vectoring_clipped = mask & (
        (raw_target < angle_lower - 1.0e-8)
        | (raw_target > angle_upper + 1.0e-8)
    )
    thrust = torch.where(mask, thrust, zero)
    target = torch.where(mask, target, current_angle)

    applied_channels = torch.stack(
        (thrust * torch.sin(target), thrust * torch.cos(target)), dim=-1
    ).reshape(batch_size, rotor_count * 2)
    achieved = (
        allocation_matrix @ applied_channels.unsqueeze(-1)
    ).squeeze(-1)
    residual = desired_wrench_body - achieved
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    # ``channels`` has already been projected onto every hard thrust,
    # vectoring-angle, vectoring-rate, and rotor-mask constraint, and
    # ``residual_norm`` is recomputed from the commands that will actually be
    # sent to the actuators.  Therefore physical allocation feasibility is
    # exactly the finite applied-wrench residual test below.  ADMM convergence
    # is an optimizer-health/optimality diagnostic; treating its tighter
    # internal stopping criterion as physical infeasibility produced false
    # QPID terminals even when the applied command was inside all constraints
    # and below the controller's supported-wrench tolerance.
    feasible = torch.isfinite(residual_norm) & (
        residual_norm <= float(unsupported_wrench_tolerance)
    )
    qp_residual = allocation_matrix @ z_value.unsqueeze(-1)
    qp_residual = qp_residual.squeeze(-1) - desired_wrench_body
    smooth = z_value - previous_flat
    objective = (
        0.5 * qp_residual.square().sum(dim=-1)
        + 0.5 * resolved.regularization_weight * z_value.square().sum(dim=-1)
        + 0.5
        * resolved.previous_command_weight
        * smooth.square().sum(dim=-1)
    )
    return BatchedVirtualThrustQPResult(
        rotor_thrusts_n=thrust,
        vectoring_joint_targets_rad=target,
        virtual_channel_solution=channels,
        achieved_wrench_body=achieved,
        residual_wrench_body=residual,
        residual_norm=residual_norm,
        feasible=feasible,
        solver_converged=solver_converged,
        thrust_clipped=thrust_clipped,
        vectoring_clipped=vectoring_clipped,
        primal_residual_norm=primal_norm,
        dual_residual_norm=dual_norm,
        objective=objective,
    )


def _project_virtual_channels(
    values: torch.Tensor,
    *,
    minimum_virtual_z: torch.Tensor,
    maximum_virtual_z: torch.Tensor,
    maximum_virtual_x: torch.Tensor,
    angle_lower: torch.Tensor,
    angle_upper: torch.Tensor,
    iterations: int,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> torch.Tensor:
    """Project exactly onto the scalar allocator's 2-D convex polytope.

    The former implementation ran ``iterations`` Dykstra sweeps for every
    ADMM iteration.  That is efficient for a very large CUDA batch, but its
    thousands of tiny kernels dominate one-to-eight-environment validation.
    In two dimensions the closest point is either the input itself, a
    perpendicular projection onto one feasible edge, or a feasible vertex.
    Enumerating those fixed candidates gives the same Euclidean projection in
    one tensor pass.  ``iterations`` remains an accepted argument so existing
    configuration hashes and callers do not change; it is validated upstream
    but no longer controls approximation quality.
    """

    del iterations
    tangent_upper = torch.tan(angle_upper)
    tangent_lower = torch.tan(angle_lower)
    zero = torch.zeros_like(maximum_virtual_x)
    one = torch.ones_like(maximum_virtual_x)
    constraints = torch.stack(
        (
            torch.stack((one, zero), dim=-1),
            torch.stack((-one, zero), dim=-1),
            torch.stack((zero, one), dim=-1),
            torch.stack((zero, -one), dim=-1),
            torch.stack((one, -tangent_upper), dim=-1),
            torch.stack((-one, tangent_lower), dim=-1),
        ),
        dim=-2,
    )
    bounds = torch.stack(
        (
            maximum_virtual_x,
            maximum_virtual_x,
            maximum_virtual_z,
            -minimum_virtual_z,
            zero,
            zero,
        ),
        dim=-1,
    )
    dtype_epsilon = torch.finfo(values.dtype).eps
    feasibility_tolerance = 64.0 * dtype_epsilon * (
        1.0 + bounds.abs().amax(dim=-1, keepdim=True)
    )

    # Candidate zero is the original point.  The next six candidates are its
    # orthogonal projections onto each supporting line.
    line_value = torch.einsum("...id,...d->...i", constraints, values)
    line_norm_squared = constraints.square().sum(dim=-1).clamp_min(dtype_epsilon)
    line_offset = (line_value - bounds) / line_norm_squared
    edge_candidates = values.unsqueeze(-2) - line_offset.unsqueeze(-1) * constraints

    # The remaining candidates are all 15 pairwise line intersections.  A
    # parallel pair is marked invalid before the common feasibility test.
    first = constraints.index_select(-2, pair_i)
    second = constraints.index_select(-2, pair_j)
    first_bound = bounds.index_select(-1, pair_i)
    second_bound = bounds.index_select(-1, pair_j)
    determinant = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    determinant_valid = determinant.abs() > (32.0 * dtype_epsilon)
    safe_determinant = torch.where(determinant_valid, determinant, one.unsqueeze(-1))
    vertices = torch.stack(
        (
            (
                first_bound * second[..., 1]
                - first[..., 1] * second_bound
            )
            / safe_determinant,
            (
                first[..., 0] * second_bound
                - first_bound * second[..., 0]
            )
            / safe_determinant,
        ),
        dim=-1,
    )
    candidates = torch.cat(
        (values.unsqueeze(-2), edge_candidates, vertices), dim=-2
    )
    constraint_values = torch.einsum(
        "...jd,...id->...ji", candidates, constraints
    )
    feasible = (
        constraint_values
        <= bounds.unsqueeze(-2) + feasibility_tolerance.unsqueeze(-2)
    ).all(dim=-1)
    feasible[..., 7:] &= determinant_valid
    distances = (candidates - values.unsqueeze(-2)).square().sum(dim=-1)
    distances = torch.where(
        feasible,
        distances,
        torch.full_like(distances, torch.inf),
    )
    selected = distances.argmin(dim=-1)
    projected = torch.gather(
        candidates,
        -2,
        selected[..., None, None].expand(*selected.shape, 1, 2),
    ).squeeze(-2)
    return projected


def _admm_convergence(
    *,
    x_value: torch.Tensor,
    z_value: torch.Tensor,
    previous_z: torch.Tensor,
    penalty: float,
    rotor_count: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    primal_norm = torch.linalg.vector_norm(x_value - z_value, dim=-1)
    dual_norm = penalty * torch.linalg.vector_norm(z_value - previous_z, dim=-1)
    scale = torch.maximum(
        torch.linalg.vector_norm(x_value, dim=-1),
        torch.linalg.vector_norm(z_value, dim=-1),
    )
    tolerance = absolute_tolerance * math.sqrt(rotor_count * 2) + (
        relative_tolerance * scale
    )
    return primal_norm, dual_norm, (primal_norm <= tolerance) & (
        dual_norm <= tolerance
    )


def _project_virtual_channels_compiled_small_batch(
    values: torch.Tensor,
    **kwargs: torch.Tensor | int,
) -> torch.Tensor:
    """Run the exact projection as one fused graph for tiny CUDA batches."""

    global _compiled_small_batch_projection
    global _small_batch_compilation_disabled
    if _small_batch_compilation_disabled:
        return _project_virtual_channels(values, **kwargs)
    if _compiled_small_batch_projection is None:
        try:
            _compiled_small_batch_projection = torch.compile(
                _project_virtual_channels,
                fullgraph=True,
                dynamic=False,
                mode="reduce-overhead",
            )
        except Exception:
            _small_batch_compilation_disabled = True
            return _project_virtual_channels(values, **kwargs)
    try:
        return _compiled_small_batch_projection(values, **kwargs)
    except Exception:
        # Exact eager projection is the semantic reference and safe fallback.
        # Retry it so genuine input/geometry errors still propagate rather
        # than being mistaken for a compilation availability problem.
        _small_batch_compilation_disabled = True
        return _project_virtual_channels(values, **kwargs)


def _validate_inputs(**values: torch.Tensor | None) -> None:
    desired = values["desired_wrench_body"]
    x_columns = values["virtual_x_wrench_columns"]
    z_columns = values["virtual_z_wrench_columns"]
    assert desired is not None and x_columns is not None and z_columns is not None
    if desired.ndim != 2 or desired.shape[-1] != 6:
        raise ValueError("desired_wrench_body must have shape [batch, 6]")
    if x_columns.ndim != 3 or x_columns.shape[-1] != 6:
        raise ValueError("virtual_x_wrench_columns must have shape [batch, rotor, 6]")
    if z_columns.shape != x_columns.shape:
        raise ValueError("virtual x/z wrench columns must have identical shape")
    batch_size, rotor_count, _ = x_columns.shape
    if desired.shape[0] != batch_size:
        raise ValueError("batched QP input batch dimensions differ")
    if not desired.is_floating_point():
        raise ValueError("batched QP tensors must use floating point")
    if desired.device != x_columns.device or desired.dtype != x_columns.dtype:
        raise ValueError("batched QP primary tensors must share device and dtype")
    expected = (batch_size, rotor_count)
    for name, value in values.items():
        if name in {
            "desired_wrench_body",
            "virtual_x_wrench_columns",
            "virtual_z_wrench_columns",
            "rotor_mask",
        }:
            continue
        assert value is not None
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    rotor_mask = values.get("rotor_mask")
    if rotor_mask is not None and rotor_mask.shape != expected:
        raise ValueError(f"rotor_mask must have shape {expected}")
    finite_values = [desired, x_columns, z_columns]
    finite_values.extend(
        value
        for name, value in values.items()
        if name not in {
            "desired_wrench_body",
            "virtual_x_wrench_columns",
            "virtual_z_wrench_columns",
            "rotor_mask",
        }
        and value is not None
    )
    if any(not bool(torch.isfinite(value).all()) for value in finite_values):
        raise ValueError("batched QP inputs must be finite")
    thrust_min = values["thrust_min_n"]
    thrust_max = values["thrust_max_n"]
    angle_lower = values["vectoring_lower_rad"]
    angle_upper = values["vectoring_upper_rad"]
    velocity = values["vectoring_velocity_limit_radps"]
    assert thrust_min is not None and thrust_max is not None
    assert angle_lower is not None and angle_upper is not None and velocity is not None
    if bool((thrust_min < 0.0).any()) or bool((thrust_max < thrust_min).any()):
        raise ValueError("batched QP thrust bounds are invalid")
    if bool((angle_upper < angle_lower).any()) or bool((velocity < 0.0).any()):
        raise ValueError("batched QP vectoring limits are invalid")


__all__ = [
    "BatchedVirtualThrustQPConfig",
    "BatchedVirtualThrustQPResult",
    "solve_batched_virtual_thrust_qp",
]
