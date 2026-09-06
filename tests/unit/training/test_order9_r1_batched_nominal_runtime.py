from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
from amsrr.simulation.order9_tensor_object_task import Order9TensorObjectTaskTarget
from amsrr.training import order9_r1_batched_nominal_runtime as runtime_module


class _FakeReference:
    def __init__(
        self,
        *,
        phase_trajectories,
        module_ids,
        joint_ids,
        device,
        dtype,
        provenance,
    ) -> None:
        del module_ids, joint_ids
        self.case = int(phase_trajectories)
        self.device = torch.device(device)
        self.dtype = dtype
        self.provenance = dict(provenance)
        self.phase_durations_s = {
            phase.value: float(self.case + phase_index + 1)
            for phase_index, phase in enumerate(ORDER9_OBJECT_TASK_PHASES)
        }
        self._references = {}
        for phase_index, phase in enumerate(ORDER9_OBJECT_TASK_PHASES):
            end = self.phase_durations_s[phase.value]
            offset = float(10 * self.case + phase_index)
            self._references[phase.value] = SimpleNamespace(
                times_s=torch.tensor([0.0, end], device=self.device, dtype=dtype),
                body_pose_world=torch.tensor(
                    [[offset, 0, 0, 0, 0, 0, 1], [offset + 2, 0, 2, 0, 0, 0, 1]],
                    device=self.device,
                    dtype=dtype,
                ),
                body_twist_world=torch.zeros((2, 6), device=self.device, dtype=dtype),
                joint_positions_rad=torch.full(
                    (2, 1, 1), offset, device=self.device, dtype=dtype
                ),
                joint_velocities_radps=torch.zeros(
                    (2, 1, 1), device=self.device, dtype=dtype
                ),
                object_pose_world=torch.tensor(
                    [[0, offset, 0, 0, 0, 0, 1], [0, offset + 2, 2, 0, 0, 0, 1]],
                    device=self.device,
                    dtype=dtype,
                ),
                object_twist_world=torch.zeros((2, 6), device=self.device, dtype=dtype),
            )

    def phase_start_reference(self, phase=ORDER9_OBJECT_TASK_PHASES[0]):
        row = self._references[phase.value]
        return SimpleNamespace(
            body_pose_local=row.body_pose_world[0].clone(),
            body_twist=row.body_twist_world[0].clone(),
            joint_positions_rad=row.joint_positions_rad[0].clone(),
            joint_velocities_radps=row.joint_velocities_radps[0].clone(),
            object_pose_local=row.object_pose_world[0].clone(),
            object_twist=row.object_twist_world[0].clone(),
        )

    def phase_reset_references(self, **kwargs):
        return kwargs


def test_batched_nominal_keeps_environment_wise_paths_and_maximum_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_module, "Order9C3NominalTensorReference", _FakeReference
    )
    bundles = [
        SimpleNamespace(phase_trajectories=index, provenance={"case": index})
        for index in (0, 1)
    ]
    runtime_module.configure_order9_r1_batched_nominal_runtime(
        bundles=bundles,
        tasks=[SimpleNamespace(), SimpleNamespace()],
        wrench_trajectories=[SimpleNamespace(), SimpleNamespace()],
        replay_count=2,
    )
    reference = runtime_module.Order9R1BatchedNominalTensorReference(
        phase_trajectories=None,
        module_ids=(0,),
        joint_ids=("joint",),
        device="cpu",
    )

    assert reference.phase_durations_s == {
        phase.value: float(phase_index + 2)
        for phase_index, phase in enumerate(ORDER9_OBJECT_TASK_PHASES)
    }
    start = reference.batched_phase_start_reference()
    assert start["body_pose_local"][:, 0].tolist() == [0.0, 0.0, 10.0, 10.0]
    sampled = reference._sample(
        reference._banks[ORDER9_OBJECT_TASK_PHASES[0].value],
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0.5, 0.5, 1.0, 1.0]),
    )
    assert torch.allclose(sampled[0][:, 0], torch.tensor([1.0, 1.0, 11.0, 11.0]))


def test_batched_nominal_dilates_lift_without_changing_bundle_provenance(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_module, "Order9C3NominalTensorReference", _FakeReference
    )
    bundle = SimpleNamespace(phase_trajectories=0, provenance={"case": 0})
    runtime_module.configure_order9_r1_batched_nominal_runtime(
        bundles=[bundle],
        tasks=[SimpleNamespace()],
        wrench_trajectories=[SimpleNamespace()],
        replay_count=1,
        phase_time_dilations={"lift": 3.0},
    )
    reference = runtime_module.Order9R1BatchedNominalTensorReference(
        phase_trajectories=None,
        module_ids=(0,),
        joint_ids=("joint",),
        device="cpu",
    )
    target = Order9TensorObjectTaskTarget(
        desired_robot_root_pose_world=torch.zeros((1, 7)),
        desired_robot_root_twist_world=torch.zeros((1, 6)),
        nominal_joint_positions_rad=torch.zeros((1, 1, 1)),
        nominal_joint_velocities_radps=torch.zeros((1, 1, 1)),
        desired_object_pose_world=torch.zeros((1, 7)),
        phase_goal_robot_root_pose_world=torch.zeros((1, 7)),
        phase_goal_object_pose_world=torch.zeros((1, 7)),
        phase_progress=torch.zeros(1),
        contact_schedule_index=torch.zeros(1, dtype=torch.long),
    )

    conditioned = reference.condition(
        target,
        phase_index=torch.tensor([2]),
        phase_elapsed_s=torch.tensor([1.5]),
        scene_origins=torch.zeros((1, 3)),
    )

    assert reference.provenance == {"case": 0}
    assert conditioned.desired_robot_root_pose_world[0, 0].item() == pytest.approx(
        2.0 + (2.0 / 6.0)
    )
    assert conditioned.phase_progress.item() == pytest.approx(1.0 / 6.0)


def test_batched_nominal_scales_lift_reference_and_goal_together(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module, "Order9C3NominalTensorReference", _FakeReference
    )
    bundle = SimpleNamespace(phase_trajectories=0, provenance={"case": 0})
    runtime_module.configure_order9_r1_batched_nominal_runtime(
        bundles=[bundle],
        tasks=[SimpleNamespace()],
        wrench_trajectories=[SimpleNamespace()],
        replay_count=1,
        maintain_vertical_scale=0.5,
    )
    reference = runtime_module.Order9R1BatchedNominalTensorReference(
        phase_trajectories=None,
        module_ids=(0,),
        joint_ids=("joint",),
        device="cpu",
    )
    target = Order9TensorObjectTaskTarget(
        desired_robot_root_pose_world=torch.zeros((1, 7)),
        desired_robot_root_twist_world=torch.zeros((1, 6)),
        nominal_joint_positions_rad=torch.zeros((1, 1, 1)),
        nominal_joint_velocities_radps=torch.zeros((1, 1, 1)),
        desired_object_pose_world=torch.zeros((1, 7)),
        phase_goal_robot_root_pose_world=torch.zeros((1, 7)),
        phase_goal_object_pose_world=torch.zeros((1, 7)),
        phase_progress=torch.zeros(1),
        contact_schedule_index=torch.zeros(1, dtype=torch.long),
    )

    conditioned = reference.condition(
        target,
        phase_index=torch.tensor([2]),
        phase_elapsed_s=torch.tensor([3.0]),
        scene_origins=torch.zeros((1, 3)),
    )

    assert conditioned.desired_robot_root_pose_world[0, 2].item() == pytest.approx(1.0)
    assert conditioned.desired_object_pose_world[0, 2].item() == pytest.approx(1.0)
    assert conditioned.phase_goal_robot_root_pose_world[0, 2].item() == pytest.approx(
        1.0
    )
    assert conditioned.phase_goal_object_pose_world[0, 2].item() == pytest.approx(1.0)
