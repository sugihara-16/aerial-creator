from __future__ import annotations

"""GPU-resident Order 9 load-limited contact preload.

This is the batched equivalent of :mod:`load_limited_contact_preload`.  It is
controller-side authority: learned ``pi_L`` still produces its ordinary
command, then this stage advances the previous absolute Dock targets at the
proven Order 8 speed until every assigned kinematic branch reaches its load
dwell.  No contact wrench is exposed to the actor.
"""

from dataclasses import replace

import torch

from amsrr.controllers.load_limited_contact_preload import (
    LOAD_LIMITED_CONTACT_PRELOAD_VERSION,
    LoadLimitedContactPreloadConfig,
)
from amsrr.policies.order9_tensor_command_decoder import (
    Order9TensorPolicyCommand,
)


ORDER9_BATCHED_CONTACT_PRELOAD_VERSION = (
    "order9_batched_nominal_end_or_contact_start_load_or_pi_h_upper_normal_wrench_dwell_v4:"
    + LOAD_LIMITED_CONTACT_PRELOAD_VERSION
)


class BatchedLoadLimitedContactPreload:
    """Stateful tensor preload for one fixed-topology Isaac bucket."""

    def __init__(
        self,
        *,
        batch_size: int,
        closure_direction_rad: torch.Tensor,
        joint_owner_mask_by_anchor: torch.Tensor,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        config: LoadLimitedContactPreloadConfig | None = None,
    ) -> None:
        self.config = config or LoadLimitedContactPreloadConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        if batch_size < 1:
            raise ValueError("batched contact preload batch size must be positive")
        direction = closure_direction_rad.to(device=self.device, dtype=dtype)
        owners = joint_owner_mask_by_anchor.to(
            device=self.device, dtype=torch.bool
        )
        if direction.ndim != 2:
            raise ValueError("contact preload closure direction must be [M, J]")
        if owners.ndim != 3 or tuple(owners.shape[1:]) != tuple(direction.shape):
            raise ValueError("contact preload owner mask must be [A, M, J]")
        if owners.shape[0] < 2:
            raise ValueError("contact preload requires at least two anchors")
        moving = direction.abs() > self.config.motion_epsilon_rad_s
        owners = owners & moving.unsqueeze(0)
        if not bool(owners.flatten(1).any(dim=1).all()):
            raise ValueError("every contact preload anchor needs a moving joint")
        peak = direction.abs().amax()
        if not bool(peak > self.config.motion_epsilon_rad_s):
            raise ValueError("contact preload closure direction cannot be zero")
        self.batch_size = int(batch_size)
        self.module_count, self.joint_count = direction.shape
        self.anchor_count = int(owners.shape[0])
        self.owner_mask = owners
        self.joint_has_owner = owners.any(dim=0)
        self.closure_velocity_rad_s = (
            direction
            * (float(self.config.maximum_speed_rad_s) / float(peak.item()))
        )
        self.initialized = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.bool
        )
        self.complete = torch.zeros_like(self.initialized)
        self.frozen_anchor = torch.zeros(
            (self.batch_size, self.anchor_count),
            device=self.device,
            dtype=torch.bool,
        )
        self.contact_start_dwell_s = torch.zeros(
            (self.batch_size, self.anchor_count),
            device=self.device,
            dtype=dtype,
        )
        self.load_dwell_s = torch.zeros_like(self.contact_start_dwell_s)
        self.normal_wrench_dwell_s = torch.zeros_like(
            self.contact_start_dwell_s
        )
        self.load_nm_by_anchor = torch.zeros_like(self.contact_start_dwell_s)
        self.normal_force_n_by_anchor = torch.zeros_like(
            self.contact_start_dwell_s
        )
        self.position_targets_rad = torch.zeros(
            (self.batch_size, self.module_count, self.joint_count),
            device=self.device,
            dtype=dtype,
        )
        self.velocity_targets_rad_s = torch.zeros_like(
            self.position_targets_rad
        )

    def reset(self, environment_mask: torch.Tensor | None = None) -> None:
        mask = (
            torch.ones_like(self.initialized)
            if environment_mask is None
            else environment_mask.to(device=self.device, dtype=torch.bool)
        )
        if tuple(mask.shape) != (self.batch_size,):
            raise ValueError("contact preload reset mask must be [B]")
        self.initialized[mask] = False
        self.complete[mask] = False
        self.frozen_anchor[mask] = False
        self.contact_start_dwell_s[mask] = 0.0
        self.load_dwell_s[mask] = 0.0
        self.normal_wrench_dwell_s[mask] = 0.0
        self.load_nm_by_anchor[mask] = 0.0
        self.normal_force_n_by_anchor[mask] = 0.0
        self.position_targets_rad[mask] = 0.0
        self.velocity_targets_rad_s[mask] = 0.0

    def apply(
        self,
        *,
        policy_command: Order9TensorPolicyCommand,
        runtime_phase_index: torch.Tensor,
        phase_progress: torch.Tensor,
        selected_contact_forces_world: torch.Tensor,
        selected_contact_mask: torch.Tensor,
        selected_signed_normal_force_n: torch.Tensor,
        required_signed_normal_force_n: torch.Tensor,
        damping_compensated_joint_load_nm: torch.Tensor,
        dt_s: float,
        contact_phase_index: int,
        lift_phase_index: int,
        transport_phase_index: int,
        place_phase_index: int,
        release_phase_index: int,
    ) -> Order9TensorPolicyCommand:
        """Return the command after deterministic preload/hold authority."""

        batch = self.batch_size
        expected = {
            "runtime_phase_index": (batch,),
            "phase_progress": (batch,),
            "selected_contact_forces_world": (batch, self.anchor_count, 3),
            "selected_contact_mask": (batch, self.anchor_count),
            "selected_signed_normal_force_n": (
                batch,
                self.anchor_count,
            ),
            "required_signed_normal_force_n": (
                batch,
                self.anchor_count,
            ),
            "damping_compensated_joint_load_nm": (
                batch,
                self.module_count,
                self.joint_count,
            ),
            "joint_position_targets_rad": (
                batch,
                self.module_count,
                self.joint_count,
            ),
        }
        values = {
            "runtime_phase_index": runtime_phase_index,
            "phase_progress": phase_progress,
            "selected_contact_forces_world": selected_contact_forces_world,
            "selected_contact_mask": selected_contact_mask,
            "selected_signed_normal_force_n": (
                selected_signed_normal_force_n
            ),
            "required_signed_normal_force_n": (
                required_signed_normal_force_n
            ),
            "damping_compensated_joint_load_nm": (
                damping_compensated_joint_load_nm
            ),
            "joint_position_targets_rad": (
                policy_command.joint_position_targets_rad
            ),
        }
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        dt = float(dt_s)
        if dt <= 0.0:
            raise ValueError("contact preload dt must be positive")
        if not bool(
            torch.isfinite(phase_progress).all()
            and (phase_progress >= 0.0).all()
            and (phase_progress <= 1.0).all()
            and torch.isfinite(selected_signed_normal_force_n).all()
            and torch.isfinite(required_signed_normal_force_n).all()
            and (selected_signed_normal_force_n >= 0.0).all()
            and (required_signed_normal_force_n >= 0.0).all()
        ):
            raise ValueError(
                "contact preload normal-force evidence must be finite and "
                "non-negative"
            )
        phase = runtime_phase_index.long()
        contact = phase == int(contact_phase_index)
        carried = (
            (phase == int(lift_phase_index))
            | (phase == int(transport_phase_index))
            | (phase == int(place_phase_index))
        )
        release_or_later = phase >= int(release_phase_index)
        inactive_before_contact = phase < int(contact_phase_index)
        self.reset(release_or_later | inactive_before_contact)

        # A phase-specific lift/transport/place reset represents an already
        # admitted preload state.  Preserve its current controller target.
        restored_hold = carried & ~self.initialized
        if bool(restored_hold.any()):
            self.initialized[restored_hold] = True
            self.complete[restored_hold] = True
            self.frozen_anchor[restored_hold] = True
            self.position_targets_rad[restored_hold] = (
                policy_command.joint_position_targets_rad[restored_hold]
            )

        force_norm = torch.linalg.vector_norm(
            selected_contact_forces_world, dim=-1
        )
        contact_present = (
            selected_contact_mask
            & (
                force_norm
                >= float(self.config.contact_start_force_threshold_n)
            )
        )
        waiting = contact & ~self.initialized
        if bool(waiting.any()):
            dwell = torch.where(
                contact_present,
                self.contact_start_dwell_s + dt,
                torch.zeros_like(self.contact_start_dwell_s),
            )
            self.contact_start_dwell_s[waiting] = dwell[waiting]
            contact_dwell_complete = (
                self.contact_start_dwell_s + 1.0e-7
                >= float(self.config.contact_start_dwell_s)
            ).all(dim=1)
            # The nominal contact trajectory normally creates the first raw
            # contact.  Under allowed physical randomization it can instead
            # finish with a sub-contact gap.  At that point the scalar Order 8
            # controller would already have been started by its caller; arm
            # this batched equivalent from the final policy target so the
            # load-limited closure can bridge the gap.  Raw-contact dwell may
            # still arm it earlier, and no actor receives contact truth.
            nominal_contact_complete = phase_progress >= (1.0 - 1.0e-6)
            arm = waiting & (
                contact_dwell_complete | nominal_contact_complete
            )
            if bool(arm.any()):
                self.initialized[arm] = True
                self.position_targets_rad[arm] = (
                    policy_command.joint_position_targets_rad[arm]
                )

        active = self.initialized & (contact | carried) & ~self.complete
        if bool(active.any()):
            loads = damping_compensated_joint_load_nm.clamp_min(0.0)
            expanded = loads.unsqueeze(1).expand(
                -1, self.anchor_count, -1, -1
            )
            masked = torch.where(
                self.owner_mask.unsqueeze(0),
                expanded,
                torch.full_like(expanded, -torch.inf),
            )
            anchor_load = masked.amax(dim=(-1, -2))
            self.load_nm_by_anchor[active] = anchor_load[active]
            above = anchor_load >= float(self.config.load_threshold_nm)
            next_dwell = torch.where(
                above & ~self.frozen_anchor,
                self.load_dwell_s + dt,
                torch.zeros_like(self.load_dwell_s),
            )
            self.load_dwell_s[active] = next_dwell[active]
            self.normal_force_n_by_anchor[active] = (
                selected_signed_normal_force_n[active]
            )
            wrench_floor_met = (
                selected_contact_mask
                & (required_signed_normal_force_n > 0.0)
                & (
                    selected_signed_normal_force_n + 1.0e-7
                    >= required_signed_normal_force_n
                )
            )
            next_wrench_dwell = torch.where(
                wrench_floor_met & ~self.frozen_anchor,
                self.normal_wrench_dwell_s + dt,
                torch.zeros_like(self.normal_wrench_dwell_s),
            )
            self.normal_wrench_dwell_s[active] = next_wrench_dwell[active]
            freeze = active.unsqueeze(1) & (
                (
                    self.load_dwell_s + 1.0e-7
                    >= float(self.config.load_dwell_s)
                )
                | (
                    self.normal_wrench_dwell_s + 1.0e-7
                    >= float(self.config.load_dwell_s)
                )
            )
            self.frozen_anchor |= freeze
            self.complete |= self.initialized & self.frozen_anchor.all(dim=1)

            any_frozen_owner = (
                self.frozen_anchor[:, :, None, None]
                & self.owner_mask[None, :, :, :]
            ).any(dim=1)
            velocity = self.closure_velocity_rad_s.unsqueeze(0).expand(
                batch, -1, -1
            )
            velocity = torch.where(
                self.joint_has_owner.unsqueeze(0) & ~any_frozen_owner,
                velocity,
                torch.zeros_like(velocity),
            )
            velocity = torch.where(
                active[:, None, None], velocity, torch.zeros_like(velocity)
            )
            self.velocity_targets_rad_s = velocity
            self.position_targets_rad = torch.where(
                active[:, None, None],
                self.position_targets_rad + velocity * dt,
                self.position_targets_rad,
            )

        override = self.initialized & (contact | carried)
        if not bool(override.any()):
            return policy_command
        position = torch.where(
            override[:, None, None],
            self.position_targets_rad,
            policy_command.joint_position_targets_rad,
        )
        velocity = torch.where(
            override[:, None, None],
            self.velocity_targets_rad_s,
            policy_command.joint_velocity_targets_radps,
        )
        torque = torch.where(
            override[:, None, None],
            torch.zeros_like(policy_command.joint_torque_bias_nm),
            policy_command.joint_torque_bias_nm,
        )
        return replace(
            policy_command,
            joint_position_targets_rad=position,
            joint_velocity_targets_radps=velocity,
            joint_torque_bias_nm=torque,
        )


__all__ = [
    "ORDER9_BATCHED_CONTACT_PRELOAD_VERSION",
    "BatchedLoadLimitedContactPreload",
]
