from __future__ import annotations

from dataclasses import fields
import math

import pytest
import torch

from amsrr.controllers.batched_rigid_body_model import (
    BatchedRigidBodyControlModelBuilder,
)
from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_cuda_graph_model_preserves_eager_values_and_output_lifetime(dtype, grad_enabled):
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_representative_order8_morphology(physical)
    eager = BatchedRigidBodyControlModelBuilder(morphology, physical)
    captured = BatchedRigidBodyControlModelBuilder(morphology, physical)
    captured._use_cuda_graph = True
    previous = None
    # Changing joints, poses, twists, and batch shape must use current values.
    with torch.set_grad_enabled(grad_enabled):
        for batch, offset in ((1, 0.0), (1, 0.13), (2, -0.08), (1, 0.04)):
            pose = torch.zeros(batch, eager.module_count, 7, device="cuda", dtype=dtype)
            pose[..., 5] = math.sin(offset / 2)
            pose[..., 6] = math.cos(offset / 2)
            pose[..., 0] = torch.arange(eager.module_count, device="cuda") * .3 + offset
            twist = torch.full((*pose.shape[:2], 6), offset, device="cuda", dtype=dtype)
            joints = torch.full((*pose.shape[:2], eager.local_joint_count), offset,
                                device="cuda", dtype=dtype)
            inputs = dict(module_pose_world=pose, module_twist_world=twist,
                          local_joint_positions_rad=joints)
            expected = eager.build(**inputs)
            actual = captured.build(**inputs)
            for field in fields(expected):
                left, right = getattr(expected, field.name), getattr(actual, field.name)
                if isinstance(left, torch.Tensor):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                else:
                    assert left == right
            if previous is not None:
                old, snapshots = previous
                for name, snapshot in snapshots.items():
                    torch.testing.assert_close(getattr(old, name), snapshot, rtol=0, atol=0)
            previous = actual, {f.name: getattr(actual, f.name).clone() for f in fields(actual)
                                if isinstance(getattr(actual, f.name), torch.Tensor)}
        pose[..., 0] = torch.nan
        with pytest.raises(ValueError, match="finite"):
            captured.build(**inputs)


def test_compiled_cpu_model_uses_fresh_state_and_preserves_previous_output():
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_representative_order8_morphology(physical)
    eager = BatchedRigidBodyControlModelBuilder(morphology, physical)
    compiled = BatchedRigidBodyControlModelBuilder(morphology, physical)
    compiled._use_cpu_compile = True
    previous = None
    with torch.no_grad():
        for offset in (0.0, .13, -.08):
            pose = torch.zeros(1, eager.module_count, 7)
            pose[..., 5], pose[..., 6] = math.sin(offset / 2), math.cos(offset / 2)
            pose[..., 0] = torch.arange(eager.module_count) * .3 + offset
            twist = torch.full((*pose.shape[:2], 6), offset)
            joints = torch.full((*pose.shape[:2], eager.local_joint_count), offset)
            inputs = dict(module_pose_world=pose, module_twist_world=twist,
                          local_joint_positions_rad=joints)
            expected, actual = eager.build(**inputs), compiled.build(**inputs)
            for field in fields(expected):
                left, right = getattr(expected, field.name), getattr(actual, field.name)
                if isinstance(left, torch.Tensor):
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
                else:
                    assert left == right
            if previous is not None:
                old, snapshots = previous
                for name, snapshot in snapshots.items():
                    torch.testing.assert_close(getattr(old, name), snapshot, rtol=0, atol=0)
            previous = actual, {f.name: getattr(actual, f.name).clone() for f in fields(actual)
                                if isinstance(getattr(actual, f.name), torch.Tensor)}


def _wrench_column(origin, axis, reaction):
    origin = torch.tensor(origin, dtype=torch.float64)
    axis = torch.tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    torque = torch.cross(origin, axis, dim=-1) + reaction * axis
    return torch.cat((axis, torque))


def test_batched_rigid_body_model_matches_scalar_q_conditioned_builder() -> None:
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    graph = build_representative_order8_morphology(physical_model)
    module_ids = sorted(module.module_id for module in graph.modules)
    builder = BatchedRigidBodyControlModelBuilder(graph, physical_model)
    module_pose = torch.tensor(
        [
            [
                [0.20 + 0.31 * index, -0.04 * index, 0.80, 0.0, 0.0, 0.0, 1.0]
                for index in range(len(module_ids))
            ]
        ],
        dtype=torch.float64,
    )
    module_twist = torch.tensor(
        [
            [
                [
                    0.01 * (index + 1),
                    -0.005 * index,
                    0.002,
                    0.01,
                    -0.02,
                    0.03,
                ]
                for index in range(len(module_ids))
            ]
        ],
        dtype=torch.float64,
    )
    joint_position = torch.zeros(
        (1, len(module_ids), len(builder.local_joint_ids)), dtype=torch.float64
    )
    for module_index in range(len(module_ids)):
        for joint_index, joint in enumerate(physical_model.joints):
            if joint.joint_type in {"revolute", "continuous"}:
                joint_position[0, module_index, joint_index] = (
                    0.03 * (module_index + 1) * ((joint_index % 3) - 1)
                )

    batched = builder.build(
        module_pose_world=module_pose,
        module_twist_world=module_twist,
        local_joint_positions_rad=joint_position,
    )
    states = []
    for module_index, module_id in enumerate(module_ids):
        states.append(
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=tuple(module_pose[0, module_index].tolist()),
                twist_world=module_twist[0, module_index].tolist(),
                joint_positions={
                    joint_id: float(joint_position[0, module_index, joint_index])
                    for joint_index, joint_id in enumerate(builder.local_joint_ids)
                },
                joint_velocities={joint_id: 0.0 for joint_id in builder.local_joint_ids},
            )
        )
    observation = RuntimeObservation(
        time_s=0.0,
        morphology_graph=graph,
        module_states=states,
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )
    scalar = RigidBodyControlModelBuilder().build(
        graph, physical_model, observation
    )

    torch.testing.assert_close(
        batched.body_pose_world[0],
        torch.tensor(scalar.body_pose_world, dtype=torch.float64),
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    torch.testing.assert_close(
        batched.body_twist_world[0],
        torch.tensor(scalar.body_twist_world, dtype=torch.float64),
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    torch.testing.assert_close(
        batched.total_mass_kg[0],
        torch.tensor(scalar.total_mass_kg, dtype=torch.float64),
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    torch.testing.assert_close(
        batched.inertia_body[0],
        torch.tensor(scalar.inertia_body, dtype=torch.float64),
        rtol=2.0e-9,
        atol=2.0e-10,
    )
    scalar_rotors = sorted(
        scalar.rotor_elements, key=lambda item: item.global_rotor_id
    )
    assert tuple(rotor.global_rotor_id for rotor in scalar_rotors) == tuple(
        f"module_{module_id}:{rotor_id}"
        for module_id, rotor_id in zip(
            batched.rotor_module_ids, batched.rotor_local_ids
        )
    )
    expected_x = torch.stack(
        [
            _wrench_column(
                rotor.origin_body,
                rotor.virtual_x_axis_body,
                rotor.reaction_torque_coeff_nm_per_n,
            )
            for rotor in scalar_rotors
        ]
    )
    expected_z = torch.stack(
        [
            _wrench_column(
                rotor.origin_body,
                rotor.virtual_z_axis_body,
                rotor.reaction_torque_coeff_nm_per_n,
            )
            for rotor in scalar_rotors
        ]
    )
    torch.testing.assert_close(
        batched.virtual_x_wrench_columns[0], expected_x, rtol=2.0e-9, atol=2.0e-10
    )
    torch.testing.assert_close(
        batched.virtual_z_wrench_columns[0], expected_z, rtol=2.0e-9, atol=2.0e-10
    )
    expected_angles = torch.tensor(
        [
            scalar.current_joint_positions[rotor.vectoring_joint_ids[0]]
            for rotor in scalar_rotors
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        batched.current_vectoring_angles_rad[0], expected_angles
    )


def test_batched_rigid_body_model_rejects_wrong_module_axis() -> None:
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    graph = build_representative_order8_morphology(physical_model)
    builder = BatchedRigidBodyControlModelBuilder(graph, physical_model)
    with torch.no_grad():
        try:
            builder.build(
                module_pose_world=torch.zeros((1, 1, 7)),
                module_twist_world=torch.zeros((1, 1, 6)),
                local_joint_positions_rad=torch.zeros(
                    (1, 1, builder.local_joint_count)
                ),
            )
        except ValueError as exc:
            assert "module_count" in str(exc)
        else:
            raise AssertionError("invalid batched rigid-body shape was accepted")
