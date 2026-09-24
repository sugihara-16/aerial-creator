from __future__ import annotations

"""Deployable, physics-based normal-force estimation for Order 9 anchors.

The estimator deliberately consumes no simulator contact tensor.  It removes
the PhysX/model gravity term and a slowly varying free-motion torque baseline,
then fits non-negative contact normal forces through the current grasp-point
Jacobian.  Raw contact remains available only to reward/evaluation code for
measuring estimator error.
"""

from dataclasses import dataclass
import math

import torch


ORDER9_ANCHOR_NORMAL_FORCE_ESTIMATOR_VERSION = (
    "order9_jacobian_normal_force_estimator_v5_contact_frame_projection"
)
ORDER9_ANCHOR_REACTION_DIRECTION_CONTRACT = (
    "order9_pi_h_wrench_to_robot_reaction_normal_v2_contact_frame_projection"
)


def order9_anchor_reaction_normal_world(
    *,
    contact_normal_world: torch.Tensor,
    contact_frame_pose_world: torch.Tensor,
    wrench_lower_contact: torch.Tensor,
    wrench_upper_contact: torch.Tensor,
) -> torch.Tensor:
    """Orient each normal as the reaction force acting on the robot.

    ``pi_H`` wrench bounds describe the action exerted by the anchor on the
    object.  The motor-current observer models the equal-and-opposite reaction
    on the robot, including the signed normal interval of opposing contacts.
    A zero-crossing normal interval has no identifiable direction and returns
    a zero vector so the observer fails closed with zero confidence.
    """

    if (
        contact_normal_world.ndim != 3
        or contact_normal_world.shape[-1] != 3
        or contact_frame_pose_world.shape
        != contact_normal_world.shape[:2] + (7,)
        or wrench_lower_contact.shape != contact_normal_world.shape[:2] + (6,)
        or wrench_upper_contact.shape != wrench_lower_contact.shape
    ):
        raise ValueError("Order9 anchor reaction direction shapes differ")
    normal_contact = order9_contact_normal_in_contact_frame(
        contact_normal_world=contact_normal_world,
        contact_frame_pose_world=contact_frame_pose_world,
    )
    target_force_contact = 0.5 * (
        wrench_lower_contact[..., :3] + wrench_upper_contact[..., :3]
    )
    action_outward_projection = (
        target_force_contact * normal_contact
    ).sum(dim=-1)
    action_sign = torch.sign(action_outward_projection)
    return -action_sign[..., None] * contact_normal_world


def order9_contact_normal_in_contact_frame(
    *,
    contact_normal_world: torch.Tensor,
    contact_frame_pose_world: torch.Tensor,
) -> torch.Tensor:
    """Express each authored surface normal in its contact-wrench frame.

    A contact frame is not required to align x with the surface normal.  The
    pitch-dock candidates, for example, can encode their grasp normal on y.
    """

    if (
        contact_normal_world.ndim != 3
        or contact_normal_world.shape[-1] != 3
        or contact_frame_pose_world.shape
        != contact_normal_world.shape[:2] + (7,)
    ):
        raise ValueError("Order9 contact-normal frame shapes differ")
    quaternion = contact_frame_pose_world[..., 3:7]
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    inverse = torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)
    normal = contact_normal_world / torch.linalg.vector_norm(
        contact_normal_world, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    return _quaternion_rotate(inverse, normal)


def order9_projected_inward_wrench_interval(
    *,
    wrench_lower_contact: torch.Tensor,
    wrench_upper_contact: torch.Tensor,
    contact_normal_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a contact-frame force box onto the physical inward normal."""

    if (
        wrench_lower_contact.shape != wrench_upper_contact.shape
        or wrench_lower_contact.ndim != 3
        or wrench_lower_contact.shape[-1] != 6
        or contact_normal_contact.shape
        != wrench_lower_contact.shape[:2] + (3,)
    ):
        raise ValueError("Order9 projected normal-wrench shapes differ")
    inward = -contact_normal_contact
    lower_force = wrench_lower_contact[..., :3]
    upper_force = wrench_upper_contact[..., :3]
    projected_lower = torch.where(
        inward >= 0.0,
        inward * lower_force,
        inward * upper_force,
    ).sum(dim=-1)
    projected_upper = torch.where(
        inward >= 0.0,
        inward * upper_force,
        inward * lower_force,
    ).sum(dim=-1)
    return projected_lower, projected_upper


def _quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = quaternion[..., :3]
    q_w = quaternion[..., 3:4]
    first = torch.cross(q_xyz, vector, dim=-1)
    return vector + 2.0 * torch.cross(
        q_xyz, first + q_w * vector, dim=-1
    )


def shift_link_origin_linear_jacobian_to_point(
    *,
    body_link_jacobian_world: torch.Tensor,
    body_position_world: torch.Tensor,
    point_position_world: torch.Tensor,
    joint_columns: torch.Tensor,
) -> torch.Tensor:
    """Shift a spatial link-origin Jacobian to a world-space point."""

    if (
        body_link_jacobian_world.ndim != 4
        or body_link_jacobian_world.shape[2] != 6
        or body_position_world.shape
        != body_link_jacobian_world.shape[:2] + (3,)
        or point_position_world.shape != body_position_world.shape
        or joint_columns.ndim != 1
    ):
        raise ValueError("Order9 grasp-point Jacobian shapes differ")
    selected = body_link_jacobian_world.index_select(3, joint_columns)
    linear = selected[..., :3, :]
    angular_columns = selected[..., 3:6, :].movedim(-2, -1)
    offset = point_position_world - body_position_world
    shifted_columns = linear.movedim(-2, -1) + torch.cross(
        angular_columns,
        offset.unsqueeze(-2).expand_as(angular_columns),
        dim=-1,
    )
    return shifted_columns.movedim(-2, -1)


def order9_branch_free_motion_baseline_update_mask(
    *,
    approach_mask: torch.Tensor,
    contact_acquisition_mask: torch.Tensor,
    selected_surface_distance_m: torch.Tensor,
    selected_anchor_mask: torch.Tensor,
    anchor_joint_owner_mask: torch.Tensor,
    baseline_initialized: torch.Tensor,
    contact_clearance_m: float,
) -> torch.Tensor:
    """Select joints whose non-contact torque offset remains observable.

    All joints are calibrated during approach.  During contact acquisition,
    an anchor's owned joints continue tracking the free-motion command load
    only while that anchor is outside the deployable geometric contact band.
    This captures closing-motion servo torque without learning actual contact
    reaction into the baseline.  A phase-specific reset that begins in
    contact is deliberately not calibrated this way because it has no prior
    free-motion observation.
    """

    if approach_mask.ndim != 1 or contact_acquisition_mask.shape != approach_mask.shape:
        raise ValueError("Order9 baseline phase-mask shape differs")
    batch = approach_mask.shape[0]
    if (
        selected_surface_distance_m.ndim != 2
        or selected_surface_distance_m.shape[0] != batch
        or selected_anchor_mask.shape != selected_surface_distance_m.shape
        or baseline_initialized.shape != (batch,)
    ):
        raise ValueError("Order9 baseline anchor-mask shape differs")
    if not math.isfinite(contact_clearance_m) or contact_clearance_m <= 0.0:
        raise ValueError("Order9 baseline contact clearance must be positive")
    owner = anchor_joint_owner_mask.to(dtype=torch.bool)
    if owner.ndim == 2:
        owner = owner.unsqueeze(0).expand(batch, -1, -1)
    if owner.ndim != 3 or owner.shape[:2] != selected_anchor_mask.shape:
        raise ValueError("Order9 baseline owner-mask shape differs")
    clearly_free_anchor = (
        selected_anchor_mask.to(dtype=torch.bool)
        & (selected_surface_distance_m > float(contact_clearance_m))
    )
    clearly_free_joint = (
        clearly_free_anchor[..., None] & owner
    ).any(dim=1)
    return approach_mask[:, None].expand(-1, owner.shape[-1]) | (
        contact_acquisition_mask[:, None]
        & baseline_initialized[:, None]
        & clearly_free_joint
    )


@dataclass(frozen=True)
class Order9AnchorNormalForceEstimatorConfig:
    baseline_update_alpha: float = 0.10
    force_filter_alpha: float = 0.25
    ridge_damping_m2: float = 1.0e-5
    projected_iterations: int = 12
    minimum_normal_jacobian_norm_m: float = 1.0e-3
    fit_residual_scale_nm: float = 0.25
    maximum_normal_force_n: float = 100.0

    def validate(self) -> None:
        for name in ("baseline_update_alpha", "force_filter_alpha"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"Order9 force estimator {name} must be in (0, 1]")
        for name in (
            "ridge_damping_m2",
            "minimum_normal_jacobian_norm_m",
            "fit_residual_scale_nm",
            "maximum_normal_force_n",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Order9 force estimator {name} must be positive")
        if self.projected_iterations < 1:
            raise ValueError("Order9 force estimator iterations must be positive")


@dataclass(frozen=True)
class Order9AnchorNormalForceEstimatorState:
    free_motion_torque_baseline_nm: torch.Tensor
    filtered_normal_force_n: torch.Tensor
    baseline_initialized: torch.Tensor


@dataclass(frozen=True)
class Order9AnchorNormalForceEstimate:
    normal_force_n: torch.Tensor
    confidence: torch.Tensor
    fit_residual_nm: torch.Tensor
    normal_jacobian_norm_m: torch.Tensor
    torque_residual_nm: torch.Tensor
    next_state: Order9AnchorNormalForceEstimatorState


class Order9AnchorNormalForceEstimator:
    """Estimate per-anchor compressive normal force from deployable signals."""

    version = ORDER9_ANCHOR_NORMAL_FORCE_ESTIMATOR_VERSION

    def __init__(
        self, config: Order9AnchorNormalForceEstimatorConfig | None = None
    ) -> None:
        self.config = config or Order9AnchorNormalForceEstimatorConfig()
        self.config.validate()
        self._use_cuda_graph = False
        self._cuda_call = None

    def initial_state(
        self,
        *,
        batch_size: int,
        anchor_count: int,
        joint_count: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> Order9AnchorNormalForceEstimatorState:
        if batch_size < 1 or anchor_count < 1 or joint_count < 1:
            raise ValueError("Order9 force estimator dimensions must be positive")
        return Order9AnchorNormalForceEstimatorState(
            free_motion_torque_baseline_nm=torch.zeros(
                batch_size, joint_count, device=device, dtype=dtype
            ),
            filtered_normal_force_n=torch.zeros(
                batch_size, anchor_count, device=device, dtype=dtype
            ),
            baseline_initialized=torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
        )

    @torch.no_grad()
    def step(
        self,
        *,
        applied_joint_torque_nm: torch.Tensor,
        gravity_joint_torque_nm: torch.Tensor,
        grasp_point_linear_jacobian_world: torch.Tensor,
        contact_normal_world: torch.Tensor,
        anchor_joint_owner_mask: torch.Tensor,
        selected_anchor_mask: torch.Tensor,
        baseline_update_mask: torch.Tensor,
        estimation_active_mask: torch.Tensor,
        state: Order9AnchorNormalForceEstimatorState,
    ) -> Order9AnchorNormalForceEstimate:
        values = dict(
            applied_joint_torque_nm=applied_joint_torque_nm,
            gravity_joint_torque_nm=gravity_joint_torque_nm,
            grasp_point_linear_jacobian_world=(
                grasp_point_linear_jacobian_world
            ),
            contact_normal_world=contact_normal_world,
            anchor_joint_owner_mask=anchor_joint_owner_mask,
            selected_anchor_mask=selected_anchor_mask,
            baseline_update_mask=baseline_update_mask,
            estimation_active_mask=estimation_active_mask,
            state=state,
        )
        self._validate(**values)
        if self._use_cuda_graph:
            from amsrr.utils.tensor_dataclass_graph import TensorDataclassGraph
            if self._cuda_call is None:
                self._cuda_call = TensorDataclassGraph()
            return self._cuda_call.call(self._compute, kwargs=values, configuration=repr(self.config))
        return self._compute(**values)

    def _compute(self, *, applied_joint_torque_nm, gravity_joint_torque_nm,
                 grasp_point_linear_jacobian_world, contact_normal_world,
                 anchor_joint_owner_mask, selected_anchor_mask, baseline_update_mask,
                 estimation_active_mask, state):
        cfg = self.config
        drive_minus_gravity = applied_joint_torque_nm - gravity_joint_torque_nm
        if baseline_update_mask.ndim == 1:
            baseline_update_joint_mask = baseline_update_mask[:, None].expand(
                -1, drive_minus_gravity.shape[-1]
            )
        else:
            baseline_update_joint_mask = baseline_update_mask
        initialize_from_free_motion = (
            ~state.baseline_initialized
            & baseline_update_joint_mask.all(dim=-1)
        )
        # A phase-specific reset can begin inside an already established
        # contact.  In that case gravity compensation is the only safe model
        # baseline; never learn the existing contact load into the baseline.
        initialize_without_free_motion = (
            ~state.baseline_initialized
            & estimation_active_mask
            & ~initialize_from_free_motion
        )
        baseline = state.free_motion_torque_baseline_nm.clone()
        baseline = torch.where(
            initialize_from_free_motion[:, None], drive_minus_gravity, baseline
        )
        update_existing = (
            baseline_update_joint_mask
            & state.baseline_initialized[:, None]
        )
        blended = (
            (1.0 - cfg.baseline_update_alpha) * baseline
            + cfg.baseline_update_alpha * drive_minus_gravity
        )
        baseline = torch.where(update_existing, blended, baseline)
        baseline_initialized = (
            state.baseline_initialized
            | initialize_from_free_motion
            | initialize_without_free_motion
        )
        torque_residual = drive_minus_gravity - baseline

        normal = contact_normal_world / torch.linalg.vector_norm(
            contact_normal_world, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        # Dynamics convention: M qdd + h = tau_actuator + J^T f_external.
        # The actuator residual caused by an outward contact force is therefore
        # -J^T n f.  Each row maps one non-negative normal force to joint torque.
        normal_jacobian = -torch.einsum(
            "bax,baxj->baj", normal, grasp_point_linear_jacobian_world
        )
        owner = anchor_joint_owner_mask.to(dtype=torch.bool)
        if owner.ndim == 2:
            owner = owner.unsqueeze(0).expand(normal_jacobian.shape[0], -1, -1)
        normal_jacobian = normal_jacobian.masked_fill(~owner, 0.0)
        selected = selected_anchor_mask.to(dtype=torch.bool)
        normal_jacobian = normal_jacobian.masked_fill(~selected[..., None], 0.0)
        jacobian_norm = torch.linalg.vector_norm(normal_jacobian, dim=-1)
        observable = (
            jacobian_norm >= cfg.minimum_normal_jacobian_norm_m
        ) & selected

        gram = torch.einsum(
            "baj,bcj->bac", normal_jacobian, normal_jacobian
        )
        identity = torch.eye(
            gram.shape[-1], device=gram.device, dtype=gram.dtype
        ).unsqueeze(0)
        gram = gram + cfg.ridge_damping_m2 * identity
        right = torch.einsum("baj,bj->ba", normal_jacobian, torque_residual)
        # Cyclic coordinate descent solves the tiny non-negative ridge
        # least-squares problem without a CPU/third-party QP dependency.  A
        # single global projected-gradient step is badly conditioned when one
        # anchor has a much shorter force moment arm than another; the exact
        # coordinate minimizer below normalizes every anchor by its own Gram
        # diagonal and therefore retains weak-but-valid force observations.
        force = torch.zeros_like(right)
        for _ in range(cfg.projected_iterations):
            for anchor_index in range(force.shape[-1]):
                gram_row = gram[:, anchor_index, :]
                diagonal = gram[:, anchor_index, anchor_index].clamp_min(1.0e-12)
                coupled = (gram_row * force).sum(dim=-1) - diagonal * force[
                    :, anchor_index
                ]
                coordinate = (
                    (right[:, anchor_index] - coupled) / diagonal
                ).clamp(0.0, cfg.maximum_normal_force_n)
                force[:, anchor_index] = torch.where(
                    observable[:, anchor_index],
                    coordinate,
                    torch.zeros_like(coordinate),
                )
        force = force.clamp_max(cfg.maximum_normal_force_n)
        predicted_torque = torch.einsum("baj,ba->bj", normal_jacobian, force)
        # Only the residual projected back onto each contact-normal Jacobian
        # can invalidate that anchor's force estimate.  The orthogonal torque
        # component contains ordinary posture control and loads from branches
        # that a normal-only contact model is intentionally unable to explain.
        # Penalizing its full norm would make every real grasp low-confidence.
        normalized_normal_jacobian = normal_jacobian / jacobian_norm.clamp_min(
            cfg.minimum_normal_jacobian_norm_m
        )[..., None]
        projected_fit_residual_by_anchor = torch.einsum(
            "baj,bj->ba",
            normalized_normal_jacobian,
            torque_residual - predicted_torque,
        ).abs()
        projected_fit_residual_by_anchor = (
            projected_fit_residual_by_anchor.masked_fill(~selected, 0.0)
        )
        fit_residual = projected_fit_residual_by_anchor.amax(dim=-1)
        fit_confidence = torch.exp(
            -projected_fit_residual_by_anchor
            / float(cfg.fit_residual_scale_nm)
        )
        observability_confidence = (
            jacobian_norm / (jacobian_norm + cfg.minimum_normal_jacobian_norm_m)
        )
        confidence = observability_confidence * fit_confidence
        confidence = confidence.masked_fill(~observable, 0.0)
        confidence = confidence.masked_fill(
            ~baseline_initialized[:, None], 0.0
        )

        filtered = (
            (1.0 - cfg.force_filter_alpha) * state.filtered_normal_force_n
            + cfg.force_filter_alpha * force
        )
        filtered = torch.where(
            estimation_active_mask[:, None], filtered, torch.zeros_like(filtered)
        )
        filtered = filtered.masked_fill(~selected, 0.0)
        confidence = torch.where(
            estimation_active_mask[:, None], confidence, torch.zeros_like(confidence)
        )
        return Order9AnchorNormalForceEstimate(
            normal_force_n=filtered,
            confidence=confidence,
            fit_residual_nm=fit_residual,
            normal_jacobian_norm_m=jacobian_norm,
            torque_residual_nm=torque_residual,
            next_state=Order9AnchorNormalForceEstimatorState(
                free_motion_torque_baseline_nm=baseline,
                filtered_normal_force_n=filtered,
                baseline_initialized=baseline_initialized,
            ),
        )

    @staticmethod
    def reset_state_subset(
        state: Order9AnchorNormalForceEstimatorState, env_ids: torch.Tensor
    ) -> Order9AnchorNormalForceEstimatorState:
        baseline = state.free_motion_torque_baseline_nm.clone()
        force = state.filtered_normal_force_n.clone()
        initialized = state.baseline_initialized.clone()
        baseline[env_ids] = 0.0
        force[env_ids] = 0.0
        initialized[env_ids] = False
        return Order9AnchorNormalForceEstimatorState(
            free_motion_torque_baseline_nm=baseline,
            filtered_normal_force_n=force,
            baseline_initialized=initialized,
        )

    @staticmethod
    def _validate(**values: torch.Tensor | Order9AnchorNormalForceEstimatorState) -> None:
        applied = values["applied_joint_torque_nm"]
        assert isinstance(applied, torch.Tensor)
        batch, joint_count = applied.shape
        jacobian = values["grasp_point_linear_jacobian_world"]
        assert isinstance(jacobian, torch.Tensor)
        if jacobian.ndim != 4 or jacobian.shape[0] != batch:
            raise ValueError("Order9 force estimator Jacobian shape differs")
        anchor_count = jacobian.shape[1]
        expected = {
            "gravity_joint_torque_nm": (batch, joint_count),
            "contact_normal_world": (batch, anchor_count, 3),
            "selected_anchor_mask": (batch, anchor_count),
            "estimation_active_mask": (batch,),
        }
        if jacobian.shape[2:] != (3, joint_count):
            raise ValueError("Order9 force estimator Jacobian joint shape differs")
        for name, shape in expected.items():
            value = values[name]
            assert isinstance(value, torch.Tensor)
            if value.shape != shape:
                raise ValueError(f"Order9 force estimator {name} shape differs")
        baseline_update_mask = values["baseline_update_mask"]
        assert isinstance(baseline_update_mask, torch.Tensor)
        if baseline_update_mask.shape not in {
            (batch,),
            (batch, joint_count),
        }:
            raise ValueError("Order9 force estimator baseline-mask shape differs")
        owner = values["anchor_joint_owner_mask"]
        assert isinstance(owner, torch.Tensor)
        if owner.shape not in {
            (anchor_count, joint_count),
            (batch, anchor_count, joint_count),
        }:
            raise ValueError("Order9 force estimator owner mask shape differs")
        state = values["state"]
        assert isinstance(state, Order9AnchorNormalForceEstimatorState)
        if (
            state.free_motion_torque_baseline_nm.shape != (batch, joint_count)
            or state.filtered_normal_force_n.shape != (batch, anchor_count)
            or state.baseline_initialized.shape != (batch,)
            or state.baseline_initialized.dtype != torch.bool
        ):
            raise ValueError("Order9 force estimator state shape differs")
        finite_flat = torch.cat([value.reshape(-1).to(device=applied.device)
                                 for value in values.values()
                                 if isinstance(value, torch.Tensor) and value.dtype != torch.bool])
        if not bool(torch.isfinite(finite_flat).all()):
            raise ValueError("Order9 force estimator input is non-finite")


def order9_required_anchor_normal_force_n(
    *,
    wrench_lower_contact: torch.Tensor,
    wrench_upper_contact: torch.Tensor,
    contact_normal_contact: torch.Tensor,
    wrench_bound_mask: torch.Tensor,
    selected_anchor_mask: torch.Tensor,
    estimated_payload_mass_kg: torch.Tensor,
    contact_friction: float,
    gravity_mps2: float = 9.81,
    support_safety_factor: float = 1.0,
) -> torch.Tensor:
    """Return a physical per-anchor minimum without simulator contact truth.

    A same-sign ``pi_H`` normal-wrench interval contributes its closest-to-zero
    admissible magnitude.  If the interval contains zero, the minimum instead
    comes from equal sharing of payload weight through Coulomb friction,
    ``m*g/(mu*N)``.  This is a phase-admission lower bound, not a force command.
    """

    if (
        wrench_lower_contact.shape != wrench_upper_contact.shape
        or wrench_lower_contact.ndim != 3
        or wrench_lower_contact.shape[-1] != 6
        or contact_normal_contact.shape
        != wrench_lower_contact.shape[:2] + (3,)
        or wrench_bound_mask.shape != wrench_lower_contact.shape[:2]
        or selected_anchor_mask.shape != wrench_bound_mask.shape
        or estimated_payload_mass_kg.shape != (wrench_lower_contact.shape[0],)
    ):
        raise ValueError("Order9 required normal-force input shape differs")
    for name, value in (
        ("contact_friction", contact_friction),
        ("gravity_mps2", gravity_mps2),
        ("support_safety_factor", support_safety_factor),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"Order9 required normal-force {name} must be positive")
    if bool((estimated_payload_mass_kg <= 0.0).any()):
        raise ValueError("Order9 estimated payload mass must be positive")
    selected_count = selected_anchor_mask.sum(dim=-1).clamp_min(1)
    support = (
        estimated_payload_mass_kg
        * float(gravity_mps2)
        * float(support_safety_factor)
        / (float(contact_friction) * selected_count.to(wrench_lower_contact.dtype))
    )[:, None]
    lower, upper = order9_projected_inward_wrench_interval(
        wrench_lower_contact=wrench_lower_contact,
        wrench_upper_contact=wrench_upper_contact,
        contact_normal_contact=contact_normal_contact,
    )
    interval_minimum = torch.where(
        lower > 0.0,
        lower,
        torch.where(upper < 0.0, -upper, torch.zeros_like(lower)),
    )
    required = torch.maximum(support.expand_as(interval_minimum), interval_minimum)
    return torch.where(
        selected_anchor_mask & wrench_bound_mask,
        required,
        torch.zeros_like(required),
    )
