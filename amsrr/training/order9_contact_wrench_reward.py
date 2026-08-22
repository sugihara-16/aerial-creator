from __future__ import annotations

"""Privileged contact-wrench range references for Order 9 ``pi_L`` reward."""

from dataclasses import dataclass
from typing import Sequence

import torch

from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.policies import ContactAssignment, ContactWrenchTrajectory
from amsrr.simulation.order9_tensor_object_task import (
    ORDER9_CONTACT_SCHEDULE_ATTACH,
    ORDER9_CONTACT_SCHEDULE_MAINTAIN,
)


ORDER9_CONTACT_WRENCH_REWARD_CONTRACT_VERSION = (
    "order9_privileged_contact_wrench_range_reward_v3_patch_moment_piecewise_log"
)
_SCHEDULE_COUNT = 5
_WRENCH_SIZE = 6


@dataclass(frozen=True)
class Order9TensorWrenchRange:
    """One batch of active bounds and object-following contact frames."""

    lower_contact: torch.Tensor
    upper_contact: torch.Tensor
    bound_mask: torch.Tensor
    contact_frame_pose_world: torch.Tensor
    contact_normal_world: torch.Tensor


class Order9TensorWrenchRangeReference:
    """Tensorize the same schedule-level wrench intent consumed by ``pi_L``.

    Candidate frames are stored relative to the task object's authored pose and
    moved with the measured object pose at reward time.  Measured contact truth
    is deliberately absent from this class and remains outside actor features.
    """

    contract_version = ORDER9_CONTACT_WRENCH_REWARD_CONTRACT_VERSION

    def __init__(
        self,
        *,
        trajectory: ContactWrenchTrajectory,
        contact_candidate_set: ContactCandidateSet,
        selected_anchor_ids: Sequence[int],
        object_id: str,
        authored_object_pose_world: Sequence[float],
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        trajectory.validate()
        contact_candidate_set.validate()
        if batch_size < 1:
            raise ValueError("Order9 wrench reference batch size must be positive")
        anchor_ids = tuple(int(value) for value in selected_anchor_ids)
        if not anchor_ids or len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("Order9 wrench reference anchors must be non-empty and unique")
        if len(authored_object_pose_world) != 7:
            raise ValueError("Order9 authored object pose must be Pose7D")
        if not object_id:
            raise ValueError("Order9 wrench reference object id must be non-empty")
        self.batch_size = int(batch_size)
        self.selected_anchor_ids = anchor_ids
        self.device = torch.device(device)
        self.dtype = dtype
        candidate_by_id = {
            candidate.candidate_id: candidate
            for candidate in contact_candidate_set.candidates
        }
        lower_bank = torch.zeros(
            (_SCHEDULE_COUNT, len(anchor_ids), _WRENCH_SIZE),
            device=self.device,
            dtype=self.dtype,
        )
        upper_bank = torch.zeros_like(lower_bank)
        mask_bank = torch.zeros(
            (_SCHEDULE_COUNT, len(anchor_ids)),
            device=self.device,
            dtype=torch.bool,
        )
        relative_pose_bank = torch.zeros(
            (_SCHEDULE_COUNT, len(anchor_ids), 7),
            device=self.device,
            dtype=self.dtype,
        )
        relative_pose_bank[..., 6] = 1.0
        relative_normal_bank = torch.zeros(
            (_SCHEDULE_COUNT, len(anchor_ids), 3),
            device=self.device,
            dtype=self.dtype,
        )
        object_inverse = inverse_pose(tuple(float(value) for value in authored_object_pose_world))
        object_quaternion = torch.tensor(
            authored_object_pose_world[3:7],
            device=self.device,
            dtype=self.dtype,
        )
        inverse_object_quaternion = torch.cat(
            (-object_quaternion[:3], object_quaternion[3:4]), dim=0
        )
        for schedule_index, schedule_label in (
            (ORDER9_CONTACT_SCHEDULE_ATTACH, "attach"),
            (ORDER9_CONTACT_SCHEDULE_MAINTAIN, "maintain"),
        ):
            assignments = _assignments_for_schedule(
                trajectory,
                schedule_label=schedule_label,
                selected_anchor_ids=anchor_ids,
            )
            for anchor_index, anchor_id in enumerate(anchor_ids):
                assignment = assignments[anchor_id]
                if assignment.wrench_frame != "contact":
                    raise SchemaValidationError(
                        "Order9 privileged wrench reward requires contact-frame bounds"
                    )
                if assignment.wrench_lower is None or assignment.wrench_upper is None:
                    raise SchemaValidationError(
                        "Order9 active privileged wrench reward bounds are missing"
                    )
                candidate = candidate_by_id.get(assignment.candidate_id)
                if candidate is None:
                    raise SchemaValidationError(
                        "Order9 privileged wrench reward candidate is missing"
                    )
                if candidate.target_entity_id != object_id:
                    raise SchemaValidationError(
                        "Order9 privileged wrench reward candidate targets another entity"
                    )
                lower = torch.tensor(
                    assignment.wrench_lower, device=self.device, dtype=self.dtype
                )
                upper = torch.tensor(
                    assignment.wrench_upper, device=self.device, dtype=self.dtype
                )
                if lower.shape != (_WRENCH_SIZE,) or upper.shape != (_WRENCH_SIZE,):
                    raise SchemaValidationError(
                        "Order9 privileged wrench reward bounds must be six-dimensional"
                    )
                if bool((lower > upper).any()):
                    raise SchemaValidationError(
                        "Order9 privileged wrench reward lower bound exceeds upper"
                    )
                relative_pose = compose_pose(
                    object_inverse,
                    tuple(float(value) for value in candidate.contact_frame_world),
                )
                lower_bank[schedule_index, anchor_index] = lower
                upper_bank[schedule_index, anchor_index] = upper
                mask_bank[schedule_index, anchor_index] = True
                relative_pose_bank[schedule_index, anchor_index] = torch.tensor(
                    relative_pose, device=self.device, dtype=self.dtype
                )
                normal_world = torch.tensor(
                    candidate.normal_world,
                    device=self.device,
                    dtype=self.dtype,
                )
                normal_world = normal_world / torch.linalg.vector_norm(
                    normal_world
                ).clamp_min(1.0e-12)
                relative_normal_bank[schedule_index, anchor_index] = (
                    _quaternion_rotate(inverse_object_quaternion, normal_world)
                )
        # Contact-wrench bounds are active only for attach/maintain, but the
        # deployable release gate still needs the actual object-surface frame
        # after those bounds turn off.  Leaving inactive rows at the identity
        # pose measures clearance from the object origin and makes opposing
        # grasp anchors impossible to release simultaneously.  The maintained
        # contact is the last active assignment, so retain its geometry for
        # approach/release/retreat/settle while keeping their bound masks off.
        inactive_schedule_indices = tuple(
            index
            for index in range(_SCHEDULE_COUNT)
            if index not in {
                ORDER9_CONTACT_SCHEDULE_ATTACH,
                ORDER9_CONTACT_SCHEDULE_MAINTAIN,
            }
        )
        for schedule_index in inactive_schedule_indices:
            relative_pose_bank[schedule_index].copy_(
                relative_pose_bank[ORDER9_CONTACT_SCHEDULE_MAINTAIN]
            )
            relative_normal_bank[schedule_index].copy_(
                relative_normal_bank[ORDER9_CONTACT_SCHEDULE_MAINTAIN]
            )
        self._lower_bank = lower_bank
        self._upper_bank = upper_bank
        self._mask_bank = mask_bank
        self._relative_pose_bank = relative_pose_bank
        self._relative_normal_bank = relative_normal_bank

    @torch.no_grad()
    def resolve(
        self,
        *,
        contact_schedule_index: torch.Tensor,
        object_pose_world: torch.Tensor,
    ) -> Order9TensorWrenchRange:
        if contact_schedule_index.shape != (self.batch_size,):
            raise ValueError("Order9 wrench schedule must have shape [batch]")
        if object_pose_world.shape != (self.batch_size, 7):
            raise ValueError("Order9 wrench object pose must have shape [batch, 7]")
        schedule = contact_schedule_index.to(device=self.device, dtype=torch.long)
        if bool((schedule < 0).any()) or bool((schedule >= _SCHEDULE_COUNT).any()):
            raise ValueError("Order9 wrench schedule index is invalid")
        object_pose = object_pose_world.to(device=self.device, dtype=self.dtype)
        relative_pose = self._relative_pose_bank.index_select(0, schedule)
        pose_world = _compose_pose_tensor(object_pose[:, None, :], relative_pose)
        relative_normal = self._relative_normal_bank.index_select(0, schedule)
        object_quaternion = _normalized_quaternion(object_pose[..., 3:7])
        normal_world = _quaternion_rotate(
            object_quaternion[:, None, :].expand(
                -1, len(self.selected_anchor_ids), -1
            ),
            relative_normal,
        )
        normal_world = normal_world / torch.linalg.vector_norm(
            normal_world, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        output = Order9TensorWrenchRange(
            lower_contact=self._lower_bank.index_select(0, schedule),
            upper_contact=self._upper_bank.index_select(0, schedule),
            bound_mask=self._mask_bank.index_select(0, schedule),
            contact_frame_pose_world=pose_world,
            contact_normal_world=normal_world,
        )
        if not bool(
            torch.isfinite(output.contact_frame_pose_world).all()
            and torch.isfinite(output.contact_normal_world).all()
        ):
            raise ValueError("Order9 wrench contact geometry is non-finite")
        return output


def _assignments_for_schedule(
    trajectory: ContactWrenchTrajectory,
    *,
    schedule_label: str,
    selected_anchor_ids: tuple[int, ...],
) -> dict[int, ContactAssignment]:
    source = next(
        (
            knot
            for knot in trajectory.knots
            if any(
                assignment.schedule_state == schedule_label
                for assignment in knot.contact_assignments
            )
        ),
        trajectory.knots[-1],
    )
    selected = set(selected_anchor_ids)
    assignments = {
        assignment.anchor_id: ContactAssignment.from_dict(assignment.to_dict())
        for assignment in source.contact_assignments
        if assignment.anchor_id in selected
    }
    if set(assignments) != selected:
        raise SchemaValidationError(
            "Order9 privileged wrench reward does not cover every selected anchor"
        )
    return assignments


def _compose_pose_tensor(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_quaternion = _normalized_quaternion(left[..., 3:7])
    right_quaternion = _normalized_quaternion(right[..., 3:7])
    translated = left[..., :3] + _quaternion_rotate(
        left_quaternion, right[..., :3]
    )
    quaternion = _quaternion_multiply(left_quaternion, right_quaternion)
    return torch.cat((translated, _normalized_quaternion(quaternion)), dim=-1)


def _quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = quaternion[..., :3]
    q_w = quaternion[..., 3:4]
    first = torch.cross(q_xyz, vector, dim=-1)
    return vector + 2.0 * torch.cross(q_xyz, first + q_w * vector, dim=-1)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def _normalized_quaternion(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


__all__ = [
    "ORDER9_CONTACT_WRENCH_REWARD_CONTRACT_VERSION",
    "Order9TensorWrenchRange",
    "Order9TensorWrenchRangeReference",
]
