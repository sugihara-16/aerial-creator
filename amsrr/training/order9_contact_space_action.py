from __future__ import annotations

"""Task-generic contact-space action basis and deployable projection adapter.

The basis is computed once when a reviewed nominal trajectory is installed.
Runtime application is tensor-only: no collision query, simulator contact
truth, or iterative IK is used in the control loop.
"""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from amsrr.feasibility.articulated_reachability import (
    CentroidalPostureIKSolver,
    _apply_base_delta,
    _pose_finite_difference,
    base_pose_for_centroidal_target,
    resolve_mesh_backed_anchor_references,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_CONTACT_SPACE_ACTION_SIZE,
    ORDER9_CONTACT_FEEDBACK_FEATURE_NAMES,
    ORDER9_CONTACT_SPACE_FEATURE_NAMES,
    ORDER9_GLOBAL_ACTION_SIZE,
    ORDER9_MAX_CONTACT_SLOTS,
    Order9LowLevelPolicyConfig,
)
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import InteractionKnot
from amsrr.training.order9_virtual_contact_compression import _global_joint_limits


ORDER9_CONTACT_SPACE_ACTION_ADAPTER_VERSION = (
    "order9_contact_space_jacobian_projection_adapter_v1"
)
ORDER9_CONTACT_NORMAL_ACTION_QUANTIZATION_VERSION = (
    "order9_contact_normal_physical_quantization_v1"
)


@dataclass(frozen=True)
class Order9ContactSpaceActionConfig:
    max_contact_slots: int = ORDER9_MAX_CONTACT_SLOTS
    # Keep one morphology-independent contact-normal authority for C3.  The
    # former 10 mm limit was insufficient for the long-chain bucket 40 even
    # though the reviewed nominal remained collision feasible.  The 20 mm
    # value is still bounded downstream by the combined-joint trust region.
    normal_translation_limit_m: float = 0.020
    tangential_translation_limit_m: float = 0.002
    rotation_limit_rad: float = 0.020
    damped_inverse_lambda: float = 2.0e-2
    centroidal_finite_difference_translation_m: float = 1.0e-4
    centroidal_finite_difference_rotation_rad: float = 1.0e-4
    maximum_combined_joint_delta_rad: float = 0.15
    # Optional morphology-independent physical quantization.  The policy may
    # keep a continuous latent Gaussian for PPO replay, while the command sent
    # through the contact Jacobian is restricted to exact physical increments.
    normal_translation_quantization_step_m: float | None = None

    def __post_init__(self) -> None:
        if self.max_contact_slots < 1:
            raise ValueError("contact-space max slots must be positive")
        for name in (
            "normal_translation_limit_m",
            "tangential_translation_limit_m",
            "rotation_limit_rad",
            "damped_inverse_lambda",
            "centroidal_finite_difference_translation_m",
            "centroidal_finite_difference_rotation_rad",
            "maximum_combined_joint_delta_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"contact-space {name} must be finite and positive")
        if self.normal_translation_quantization_step_m is not None:
            step = float(self.normal_translation_quantization_step_m)
            if (
                not math.isfinite(step)
                or step <= 0.0
                or step > float(self.normal_translation_limit_m)
            ):
                raise ValueError(
                    "contact-space normal quantization step must lie in "
                    "(0, normal_translation_limit_m]"
                )


@dataclass(frozen=True)
class Order9ContactSpaceActionBasis:
    module_ids: tuple[int, ...]
    local_joint_ids: tuple[str, ...]
    contact_slot_features: torch.Tensor
    contact_slot_owner_module_indices: torch.Tensor
    contact_slot_mask: torch.Tensor
    contact_to_joint_position: torch.Tensor
    joint_nullspace_projector: torch.Tensor
    centroidal_pose_projector: torch.Tensor
    centroidal_joint_compensation: torch.Tensor
    contact_action_scale: torch.Tensor
    selected_anchor_ids: tuple[int, ...]
    selected_candidate_ids: tuple[int, ...]
    normal_translation_quantization_step_m: float | None = None
    version: str = ORDER9_CONTACT_SPACE_ACTION_ADAPTER_VERSION

    @property
    def joint_count(self) -> int:
        return len(self.module_ids) * len(self.local_joint_ids)

    def to(
        self, *, device: torch.device | str, dtype: torch.dtype
    ) -> "Order9ContactSpaceActionBasis":
        def move(value: torch.Tensor) -> torch.Tensor:
            target_dtype = dtype if value.is_floating_point() else value.dtype
            return value.to(device=device, dtype=target_dtype)

        return Order9ContactSpaceActionBasis(
            module_ids=self.module_ids,
            local_joint_ids=self.local_joint_ids,
            contact_slot_features=move(self.contact_slot_features),
            contact_slot_owner_module_indices=move(
                self.contact_slot_owner_module_indices
            ),
            contact_slot_mask=move(self.contact_slot_mask),
            contact_to_joint_position=move(self.contact_to_joint_position),
            joint_nullspace_projector=move(self.joint_nullspace_projector),
            centroidal_pose_projector=move(self.centroidal_pose_projector),
            centroidal_joint_compensation=move(
                self.centroidal_joint_compensation
            ),
            contact_action_scale=move(self.contact_action_scale),
            selected_anchor_ids=self.selected_anchor_ids,
            selected_candidate_ids=self.selected_candidate_ids,
            normal_translation_quantization_step_m=(
                self.normal_translation_quantization_step_m
            ),
        )


@dataclass(frozen=True)
class Order9ProjectedContactSpaceAction:
    global_action: torch.Tensor
    joint_action: torch.Tensor
    contact_constraint_weight: torch.Tensor
    contact_joint_delta_rad: torch.Tensor
    centroidal_compensation_joint_delta_rad: torch.Tensor
    posture_nullspace_joint_delta_rad: torch.Tensor


def apply_order9_diagnostic_contact_normal_residual(
    *,
    normalized_contact_action: torch.Tensor,
    inward_residual_m: torch.Tensor,
    basis: Order9ContactSpaceActionBasis,
) -> torch.Tensor:
    """Override active contact-normal actions for a diagnostic A/B sweep.

    This helper deliberately changes only the first (inward-normal) coordinate
    of each active contact slot.  It is not a production controller or a
    training transform; callers must mark all resulting evidence as
    acceptance-ineligible.
    """

    if normalized_contact_action.ndim != 3 or (
        normalized_contact_action.shape[1:]
        != (
            basis.contact_slot_mask.shape[0],
            ORDER9_CONTACT_SPACE_ACTION_SIZE,
        )
    ):
        raise ValueError("diagnostic contact-normal action shape differs")
    batch = normalized_contact_action.shape[0]
    if inward_residual_m.shape != (batch,):
        raise ValueError("diagnostic contact-normal residual shape differs")
    if not bool(torch.isfinite(normalized_contact_action).all()) or not bool(
        torch.isfinite(inward_residual_m).all()
    ):
        raise ValueError("diagnostic contact-normal residual is non-finite")
    device = normalized_contact_action.device
    dtype = normalized_contact_action.dtype
    moved_basis = basis.to(device=device, dtype=dtype)
    normal_limit_m = moved_basis.contact_action_scale[0]
    if bool((inward_residual_m.abs() > normal_limit_m + 1.0e-9).any()):
        raise ValueError("diagnostic contact-normal residual exceeds action span")
    normalized_residual = inward_residual_m.to(device=device, dtype=dtype) / (
        normal_limit_m
    )
    output = normalized_contact_action.clone()
    active = moved_basis.contact_slot_mask.unsqueeze(0).expand(batch, -1)
    output[:, :, 0] = torch.where(
        active,
        normalized_residual.unsqueeze(1).expand_as(output[:, :, 0]),
        output[:, :, 0],
    )
    return output


def build_order9_deployable_contact_feedback_features(
    *,
    signed_surface_distance_m: torch.Tensor,
    relative_twist_contact: torch.Tensor,
    signed_hardware_joint_load_nm: torch.Tensor,
    basis: Order9ContactSpaceActionBasis,
) -> torch.Tensor:
    """Build contact feedback available from kinematics and motor current.

    ``signed_surface_distance_m`` and ``relative_twist_contact`` describe the
    selected grasp-contact frames relative to their object-surface targets.
    The load proxy projects signed motor effort onto the same joint direction
    that realizes a positive inward-normal contact residual.  This is ordinary
    proprioception; raw simulator contact force is neither accepted nor used.
    """

    if signed_surface_distance_m.ndim != 2:
        raise ValueError("contact feedback surface distance must be [batch, contact]")
    batch, contact_count = signed_surface_distance_m.shape
    if relative_twist_contact.shape != (batch, contact_count, 6):
        raise ValueError("contact feedback relative twist shape differs")
    module_count = len(basis.module_ids)
    joint_count = len(basis.local_joint_ids)
    if signed_hardware_joint_load_nm.shape != (
        batch,
        module_count,
        joint_count,
    ):
        raise ValueError("contact feedback signed joint-load shape differs")
    if contact_count != len(basis.selected_anchor_ids):
        raise ValueError("contact feedback selected-contact count differs")
    for value in (
        signed_surface_distance_m,
        relative_twist_contact,
        signed_hardware_joint_load_nm,
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("contact feedback contains non-finite values")

    device = signed_surface_distance_m.device
    dtype = signed_surface_distance_m.dtype
    moved = basis.to(device=device, dtype=dtype)
    slot_count = int(moved.contact_slot_mask.shape[0])
    output = torch.zeros(
        (batch, slot_count, len(ORDER9_CONTACT_FEEDBACK_FEATURE_NAMES)),
        device=device,
        dtype=dtype,
    )
    normal_span = moved.contact_action_scale[0].clamp_min(1.0e-9)
    output[:, :contact_count, 0] = (
        signed_surface_distance_m / normal_span
    ).clamp(-2.0, 2.0)
    output[:, :contact_count, 1:4] = relative_twist_contact[..., :3].clamp(
        -2.0, 2.0
    )
    output[:, :contact_count, 4:7] = relative_twist_contact[..., 3:6].clamp(
        -4.0, 4.0
    )

    normal_joint_columns = moved.contact_to_joint_position[:, 0::6]
    normal_joint_columns = normal_joint_columns[:, :slot_count]
    normal_joint_directions = normal_joint_columns / torch.linalg.vector_norm(
        normal_joint_columns, dim=0, keepdim=True
    ).clamp_min(1.0e-9)
    flattened_load = signed_hardware_joint_load_nm.reshape(batch, -1)
    load_proxy = torch.einsum(
        "bj,js->bs", flattened_load, normal_joint_directions
    )
    output[:, :, 7] = torch.sign(load_proxy) * torch.log1p(load_proxy.abs())
    output *= moved.contact_slot_mask.reshape(1, slot_count, 1).to(dtype)
    return output


def build_order9_contact_space_action_basis(
    *,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    contact_knot: InteractionKnot,
    candidate_set: ContactCandidateSet,
    module_ids: Sequence[int],
    local_joint_ids: Sequence[str],
    config: Order9ContactSpaceActionConfig | None = None,
) -> Order9ContactSpaceActionBasis:
    """Linearize contact kinematics around one reviewed nominal grasp."""

    resolved = config or Order9ContactSpaceActionConfig()
    morphology.validate()
    physical_model.validate()
    candidate_set.validate()
    posture = contact_knot.posture_target
    centroidal = contact_knot.centroidal_target
    if (
        posture is None
        or posture.joint_pos_target is None
        or centroidal is None
        or centroidal.com_pos_world is None
        or centroidal.body_orientation_world is None
    ):
        raise ValueError("contact-space basis requires a resolved contact posture")
    assignments = tuple(
        sorted(
            contact_knot.contact_assignments,
            key=lambda value: (
                int(value.slot_id),
                int(value.anchor_id),
                int(value.candidate_id),
            ),
        )
    )
    if not assignments:
        raise ValueError("contact-space basis requires at least one assignment")
    if len(assignments) > resolved.max_contact_slots:
        raise ValueError("contact-space assignment count exceeds padded slots")

    ordered_ids = ordered_global_dock_joint_ids(morphology, physical_model)
    expected_ids = tuple(
        f"module_{int(module_id)}:{joint_id}"
        for module_id in module_ids
        for joint_id in local_joint_ids
    )
    if set(ordered_ids) != set(expected_ids):
        raise ValueError("contact-space runtime joint identity differs from IK")
    q = {joint_id: float(posture.joint_pos_target[joint_id]) for joint_id in ordered_ids}
    limits = _global_joint_limits(ordered_ids, physical_model)
    selected_anchor_ids = tuple(int(value.anchor_id) for value in assignments)
    references = resolve_mesh_backed_anchor_references(
        morphology, physical_model, selected_anchor_ids
    )
    solver = CentroidalPostureIKSolver(physical_model)
    centroidal_pose: Pose7D = (
        *tuple(float(value) for value in centroidal.com_pos_world),
        *tuple(float(value) for value in centroidal.body_orientation_world),
    )
    base_pose = base_pose_for_centroidal_target(
        morphology,
        physical_model,
        q,
        centroidal_pose[:3],
        centroidal_pose[3:7],
        kinematics=solver.kinematics,
    )
    nominal = solver.kinematics.forward(
        morphology, physical_model, q, base_pose, references
    )
    jacobians = solver._fixed_centroidal_anchor_jacobians(
        morphology=morphology,
        centroidal_pose_world=centroidal_pose,
        q=q,
        limits=limits,
        references=references,
        nominal=nominal.anchor_poses_world,
    )
    column_indices = [ordered_ids.index(value) for value in expected_ids]
    full_joint_jacobian = np.concatenate(
        [
            np.asarray(jacobians[anchor_id], dtype=float)[:, column_indices]
            for anchor_id in selected_anchor_ids
        ],
        axis=0,
    )
    # Contact maintenance uses the translational no-slip point constraints.
    # Requiring two complete 6-D end-effector poses would mathematically lock
    # every centroidal degree of freedom and is stronger than the physical
    # grasp contract.  Orientation remains an explicit soft contact action.
    constraint_joint_jacobian = np.concatenate(
        [
            np.asarray(jacobians[anchor_id], dtype=float)[:3, column_indices]
            for anchor_id in selected_anchor_ids
        ],
        axis=0,
    )
    full_joint_inverse = _damped_right_inverse(
        full_joint_jacobian, float(resolved.damped_inverse_lambda)
    )
    constraint_joint_inverse = np.linalg.pinv(
        constraint_joint_jacobian, rcond=1.0e-6
    )
    joint_nullspace = (
        np.eye(len(expected_ids))
        - constraint_joint_inverse @ constraint_joint_jacobian
    )

    candidate_by_id = {
        int(candidate.candidate_id): candidate
        for candidate in candidate_set.candidates
    }
    anchor_by_id = {int(anchor.anchor_id): anchor for anchor in morphology.robot_anchors}
    module_index = {int(value): index for index, value in enumerate(module_ids)}
    feature = np.zeros(
        (resolved.max_contact_slots, len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    owner = np.full((resolved.max_contact_slots,), -1, dtype=np.int64)
    mask = np.zeros((resolved.max_contact_slots,), dtype=np.bool_)
    contact_frame_map = np.zeros(
        (
            6 * len(assignments),
            resolved.max_contact_slots * ORDER9_CONTACT_SPACE_ACTION_SIZE,
        ),
        dtype=float,
    )
    selected_candidate_ids: list[int] = []
    for assignment_index, assignment in enumerate(assignments):
        # Task contact-slot ids need not be unique (a grasp pair commonly
        # belongs to one object slot).  The policy slot is therefore the
        # deterministic assignment order, not ContactAssignment.slot_id.
        slot = assignment_index
        candidate = candidate_by_id[int(assignment.candidate_id)]
        anchor = anchor_by_id[int(assignment.anchor_id)]
        if int(anchor.module_id) not in module_index:
            raise ValueError("contact-space anchor owner module is absent")
        normal = _unit(candidate.normal_world, "contact normal")
        tangent_1 = _unit(candidate.tangent_basis_world[:3], "contact tangent 1")
        tangent_2 = _unit(candidate.tangent_basis_world[3:6], "contact tangent 2")
        # Translation action is inward-normal, tangent-1, tangent-2.  Rotation
        # uses the same right-handed local axes.
        axes = np.stack((-normal, tangent_1, tangent_2), axis=1)
        local_transform = np.zeros((6, 6), dtype=float)
        local_transform[:3, :3] = axes
        local_transform[3:, 3:] = axes
        row = slice(6 * assignment_index, 6 * (assignment_index + 1))
        col = slice(
            ORDER9_CONTACT_SPACE_ACTION_SIZE * slot,
            ORDER9_CONTACT_SPACE_ACTION_SIZE * (slot + 1),
        )
        contact_frame_map[row, col] = local_transform
        wrench_target = _six_or_midpoint(
            assignment.wrench_target,
            assignment.wrench_lower,
            assignment.wrench_upper,
        )
        wrench_half_range = _six_half_range(
            assignment.wrench_lower, assignment.wrench_upper
        )
        relative = np.asarray(candidate.contact_pose_world[:3], dtype=float) - np.asarray(
            centroidal_pose[:3], dtype=float
        )
        feature[slot] = np.asarray(
            [
                *normal,
                *tangent_1,
                *tangent_2,
                *tuple(float(value) for value in anchor.local_pose[:3]),
                *relative,
                *[_signed_log(value) for value in wrench_target],
                *[math.log1p(max(0.0, value)) for value in wrench_half_range],
                float(candidate.friction or 0.0),
                float(candidate.patch_area_m2),
                float(assignment.priority),
            ],
            dtype=np.float32,
        )
        owner[slot] = module_index[int(anchor.module_id)]
        mask[slot] = True
        selected_candidate_ids.append(int(assignment.candidate_id))

    action_scale = np.asarray(
        [
            resolved.normal_translation_limit_m,
            resolved.tangential_translation_limit_m,
            resolved.tangential_translation_limit_m,
            resolved.rotation_limit_rad,
            resolved.rotation_limit_rad,
            resolved.rotation_limit_rad,
        ],
        dtype=float,
    )
    contact_to_joint = full_joint_inverse @ contact_frame_map
    full_centroidal_jacobian = _centroidal_anchor_jacobian(
        morphology=morphology,
        physical_model=physical_model,
        solver=solver,
        q=q,
        references=references,
        nominal=nominal.anchor_poses_world,
        centroidal_pose=centroidal_pose,
        translation_step=resolved.centroidal_finite_difference_translation_m,
        rotation_step=resolved.centroidal_finite_difference_rotation_rad,
    )
    centroidal_jacobian = np.concatenate(
        [
            full_centroidal_jacobian[6 * index : 6 * index + 3]
            for index in range(len(assignments))
        ],
        axis=0,
    )
    joint_achievable = constraint_joint_jacobian @ constraint_joint_inverse
    uncompensated = (
        np.eye(constraint_joint_jacobian.shape[0]) - joint_achievable
    ) @ centroidal_jacobian
    centroidal_projector = np.eye(6) - np.linalg.pinv(uncompensated, rcond=1.0e-6) @ uncompensated
    centroidal_compensation = (
        -constraint_joint_inverse @ centroidal_jacobian @ centroidal_projector
    )
    return Order9ContactSpaceActionBasis(
        module_ids=tuple(int(value) for value in module_ids),
        local_joint_ids=tuple(str(value) for value in local_joint_ids),
        contact_slot_features=torch.from_numpy(feature),
        contact_slot_owner_module_indices=torch.from_numpy(owner),
        contact_slot_mask=torch.from_numpy(mask),
        contact_to_joint_position=torch.tensor(contact_to_joint, dtype=torch.float32),
        joint_nullspace_projector=torch.tensor(joint_nullspace, dtype=torch.float32),
        centroidal_pose_projector=torch.tensor(
            centroidal_projector, dtype=torch.float32
        ),
        centroidal_joint_compensation=torch.tensor(
            centroidal_compensation, dtype=torch.float32
        ),
        contact_action_scale=torch.from_numpy(action_scale.astype(np.float32)),
        selected_anchor_ids=selected_anchor_ids,
        selected_candidate_ids=tuple(selected_candidate_ids),
        normal_translation_quantization_step_m=(
            resolved.normal_translation_quantization_step_m
        ),
    )


def apply_order9_contact_space_action(
    *,
    normalized_global_action: torch.Tensor,
    normalized_joint_action: torch.Tensor,
    normalized_contact_action: torch.Tensor,
    phase_index: torch.Tensor,
    phase_progress: torch.Tensor,
    basis: Order9ContactSpaceActionBasis,
    policy_config: Order9LowLevelPolicyConfig,
    maximum_combined_joint_delta_rad: float = 0.15,
) -> Order9ProjectedContactSpaceAction:
    """Apply the v7 contact-compatible action contract to one tensor batch."""

    if normalized_global_action.ndim != 2 or normalized_global_action.shape[1] != (
        ORDER9_GLOBAL_ACTION_SIZE
    ):
        raise ValueError("contact-space global action shape differs")
    batch = normalized_global_action.shape[0]
    modules = len(basis.module_ids)
    joints = len(basis.local_joint_ids)
    expected_joint_width = 3 * policy_config.max_local_joint_slots
    if normalized_joint_action.shape != (batch, modules, expected_joint_width):
        raise ValueError("contact-space joint action shape differs")
    if normalized_contact_action.shape != (
        batch,
        basis.contact_slot_mask.shape[0],
        ORDER9_CONTACT_SPACE_ACTION_SIZE,
    ):
        raise ValueError("contact-space residual action shape differs")
    if phase_index.shape != (batch,) or phase_progress.shape != (batch,):
        raise ValueError("contact-space phase tensor shape differs")
    for value in (
        normalized_global_action,
        normalized_joint_action,
        normalized_contact_action,
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("contact-space action contains non-finite values")
        if bool((value.abs() > 1.0 + 1.0e-6).any()):
            raise ValueError("contact-space normalized action is outside [-1, 1]")
    device, dtype = normalized_global_action.device, normalized_global_action.dtype
    basis = basis.to(device=device, dtype=dtype)
    weight = _actor_contact_constraint_weight(phase_index, phase_progress)
    eye_joint = torch.eye(modules * joints, device=device, dtype=dtype)
    eye_centroidal = torch.eye(6, device=device, dtype=dtype)
    centroidal_blend = (
        eye_centroidal.unsqueeze(0)
        + weight[:, None, None]
        * (basis.centroidal_pose_projector.unsqueeze(0) - eye_centroidal.unsqueeze(0))
    )
    joint_blend = (
        eye_joint.unsqueeze(0)
        + weight[:, None, None]
        * (basis.joint_nullspace_projector.unsqueeze(0) - eye_joint.unsqueeze(0))
    )
    pose_scale = torch.tensor(
        [
            *([float(policy_config.centroidal_position_correction_limit_m)] * 3),
            *([float(policy_config.centroidal_orientation_correction_limit_rad)] * 3),
        ],
        device=device,
        dtype=dtype,
    )
    twist_scale = torch.tensor(
        [
            *([float(policy_config.linear_twist_correction_limit_mps)] * 3),
            *([float(policy_config.angular_twist_correction_limit_radps)] * 3),
        ],
        device=device,
        dtype=dtype,
    )
    pose_physical = normalized_global_action[:, :6] * pose_scale
    twist_physical = normalized_global_action[:, 6:12] * twist_scale
    projected_pose = torch.einsum("bij,bj->bi", centroidal_blend, pose_physical)
    projected_twist = torch.einsum("bij,bj->bi", centroidal_blend, twist_physical)
    output_global = torch.zeros_like(normalized_global_action)
    output_global[:, :6] = projected_pose / pose_scale
    output_global[:, 6:12] = projected_twist / twist_scale
    output_global.clamp_(-1.0, 1.0)

    direct_q = (
        normalized_joint_action[:, :, :joints].reshape(batch, -1)
        * float(policy_config.joint_position_delta_limit_rad)
    )
    direct_qdot = (
        normalized_joint_action[
            :,
            :,
            policy_config.max_local_joint_slots : (
                policy_config.max_local_joint_slots + joints
            ),
        ].reshape(batch, -1)
        * float(policy_config.joint_velocity_limit_rad_s)
    )
    null_q = torch.einsum("bij,bj->bi", joint_blend, direct_q)
    null_qdot = torch.einsum("bij,bj->bi", joint_blend, direct_qdot)
    contact_physical_by_slot = (
        normalized_contact_action * basis.contact_action_scale.reshape(1, 1, -1)
    )
    if basis.normal_translation_quantization_step_m is not None:
        step_m = torch.as_tensor(
            basis.normal_translation_quantization_step_m,
            device=device,
            dtype=dtype,
        )
        normal_limit_m = basis.contact_action_scale[0]
        quantized_normal = (
            torch.round(contact_physical_by_slot[:, :, 0] / step_m) * step_m
        ).clamp(-normal_limit_m, normal_limit_m)
        contact_physical_by_slot = contact_physical_by_slot.clone()
        contact_physical_by_slot[:, :, 0] = quantized_normal
    contact_physical = contact_physical_by_slot.reshape(batch, -1)
    contact_delta = torch.einsum(
        "ij,bj->bi", basis.contact_to_joint_position, contact_physical
    ) * weight[:, None]
    centroidal_delta = torch.einsum(
        "ij,bj->bi", basis.centroidal_joint_compensation, projected_pose
    ) * weight[:, None]
    centroidal_qdot = torch.einsum(
        "ij,bj->bi", basis.centroidal_joint_compensation, projected_twist
    ) * weight[:, None]
    combined_q = torch.clamp(
        null_q + contact_delta + centroidal_delta,
        -float(maximum_combined_joint_delta_rad),
        float(maximum_combined_joint_delta_rad),
    )
    output_joint = torch.zeros_like(normalized_joint_action)
    output_joint[:, :, :joints] = (
        combined_q.reshape(batch, modules, joints)
        / float(policy_config.joint_position_delta_limit_rad)
    ).clamp(-1.0, 1.0)
    output_joint[
        :,
        :,
        policy_config.max_local_joint_slots : (
            policy_config.max_local_joint_slots + joints
        ),
    ] = (
        (null_qdot + centroidal_qdot).reshape(batch, modules, joints)
        / float(policy_config.joint_velocity_limit_rad_s)
    ).clamp(-1.0, 1.0)
    # Torque coordinates stay zero; QPID remains the sole torque-level path.
    return Order9ProjectedContactSpaceAction(
        global_action=output_global,
        joint_action=output_joint,
        contact_constraint_weight=weight,
        contact_joint_delta_rad=contact_delta.reshape(batch, modules, joints),
        centroidal_compensation_joint_delta_rad=centroidal_delta.reshape(
            batch, modules, joints
        ),
        posture_nullspace_joint_delta_rad=null_q.reshape(batch, modules, joints),
    )


def _centroidal_anchor_jacobian(
    *,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    solver: CentroidalPostureIKSolver,
    q: Mapping[str, float],
    references,
    nominal: Mapping[int, Pose7D],
    centroidal_pose: Pose7D,
    translation_step: float,
    rotation_step: float,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for index in range(6):
        rotation = [0.0, 0.0, 0.0]
        translation = [0.0, 0.0, 0.0]
        step = translation_step if index < 3 else rotation_step
        if index < 3:
            translation[index] = step
        else:
            rotation[index - 3] = step
        perturbed_centroidal = _apply_base_delta(
            centroidal_pose, rotation, translation
        )
        perturbed_base = base_pose_for_centroidal_target(
            morphology,
            physical_model,
            q,
            perturbed_centroidal[:3],
            perturbed_centroidal[3:7],
            kinematics=solver.kinematics,
        )
        perturbed = solver.kinematics.forward(
            morphology, physical_model, q, perturbed_base, references
        ).anchor_poses_world
        columns.append(
            np.concatenate(
                [
                    np.asarray(
                        _pose_finite_difference(
                            nominal[int(reference.anchor.anchor_id)],
                            perturbed[int(reference.anchor.anchor_id)],
                            step,
                        ),
                        dtype=float,
                    )
                    for reference in references
                ]
            )
        )
    return np.stack(columns, axis=1)


def _damped_right_inverse(matrix: np.ndarray, damping: float) -> np.ndarray:
    return matrix.T @ np.linalg.solve(
        matrix @ matrix.T + damping**2 * np.eye(matrix.shape[0]),
        np.eye(matrix.shape[0]),
    )


def _unit(values: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise ValueError(f"{label} is invalid")
    return result / norm


def _six_or_midpoint(
    target: Sequence[float] | None,
    lower: Sequence[float] | None,
    upper: Sequence[float] | None,
) -> tuple[float, ...]:
    if target is not None:
        return tuple(float(value) for value in target)
    if lower is not None and upper is not None:
        return tuple(0.5 * (float(lo) + float(hi)) for lo, hi in zip(lower, upper))
    return (0.0,) * 6


def _six_half_range(
    lower: Sequence[float] | None, upper: Sequence[float] | None
) -> tuple[float, ...]:
    if lower is None or upper is None:
        return (0.0,) * 6
    return tuple(0.5 * max(0.0, float(hi) - float(lo)) for lo, hi in zip(lower, upper))


def _signed_log(value: float) -> float:
    return math.copysign(math.log1p(abs(float(value))), float(value))


def _actor_contact_constraint_weight(
    phase_index: torch.Tensor, phase_progress: torch.Tensor
) -> torch.Tensor:
    """Constraint ramp in the actor's eleven-phase index convention."""

    progress = phase_progress.clamp(0.0, 1.0)
    smooth = progress.square() * (3.0 - 2.0 * progress)
    result = torch.zeros_like(progress)
    result = torch.where(phase_index == 1, smooth, result)
    hold = (
        (phase_index == 3)
        | (phase_index == 4)
        | (phase_index == 5)
    )
    result = torch.where(hold, torch.ones_like(result), result)
    result = torch.where(phase_index == 6, 1.0 - smooth, result)
    return result


__all__ = [
    "ORDER9_CONTACT_NORMAL_ACTION_QUANTIZATION_VERSION",
    "ORDER9_CONTACT_SPACE_ACTION_ADAPTER_VERSION",
    "Order9ContactSpaceActionBasis",
    "Order9ContactSpaceActionConfig",
    "Order9ProjectedContactSpaceAction",
    "apply_order9_contact_space_action",
    "build_order9_contact_space_action_basis",
]
