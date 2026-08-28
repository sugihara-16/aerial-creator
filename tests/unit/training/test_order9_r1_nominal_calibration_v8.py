from __future__ import annotations

import torch

from amsrr.training.order9_r1_nominal_calibration_v8 import (
    _controller_commands_are_active,
)


def test_controller_command_evidence_requires_all_three_active_tensors() -> None:
    tensors = {
        "rotor_thrusts_n": torch.ones(2, 2, 4),
        "command_joint_position_targets_rad": torch.ones(2, 2, 3),
        "controller_desired_wrench_body": torch.ones(2, 2, 6),
    }

    assert _controller_commands_are_active(tensors)
    tensors["controller_desired_wrench_body"].zero_()
    assert not _controller_commands_are_active(tensors)


def test_controller_command_evidence_rejects_nonfinite_tensor() -> None:
    tensors = {
        "rotor_thrusts_n": torch.ones(2, 2, 4),
        "command_joint_position_targets_rad": torch.ones(2, 2, 3),
        "controller_desired_wrench_body": torch.full((2, 2, 6), float("nan")),
    }

    assert not _controller_commands_are_active(tensors)
