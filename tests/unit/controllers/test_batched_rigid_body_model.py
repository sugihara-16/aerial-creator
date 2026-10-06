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


def test_articulated_com_velocity_matches_independent_pose_and_encoder_derivative():
    import numpy as np
    from scipy.spatial.transform import Rotation
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_representative_order8_morphology(physical)
    builder = BatchedRigidBodyControlModelBuilder(morphology, physical)
    rng = np.random.default_rng(912)
    count = builder.module_count
    pose = np.column_stack((rng.normal(size=(count, 3)),
        Rotation.from_rotvec(rng.normal(size=(count, 3))*.3).as_quat()))
    twist = rng.normal(size=(count, 6))*.2
    q = rng.normal(size=(count, builder.local_joint_count))*.15
    qdot = rng.normal(size=q.shape)*.3
    tensor = lambda x: torch.tensor(x[None], dtype=torch.float64)
    def evaluate(t):
        shifted = pose.copy()
        shifted[:, :3] += t*twist[:, :3]
        shifted[:, 3:] = (Rotation.from_rotvec(t*twist[:, 3:])*Rotation.from_quat(pose[:, 3:])).as_quat()
        return builder.build(module_pose_world=tensor(shifted), module_twist_world=tensor(twist),
            local_joint_positions_rad=tensor(q+t*qdot), local_joint_velocities_rad=tensor(qdot))
    epsilon = 1e-6
    derivative = (evaluate(epsilon).body_pose_world[:, :3]-evaluate(-epsilon).body_pose_world[:, :3])/(2*epsilon)
    torch.testing.assert_close(evaluate(0).body_twist_world[:, :3], derivative, atol=1e-9, rtol=1e-7)
    legacy = builder.build(module_pose_world=tensor(pose), module_twist_world=tensor(twist),
        local_joint_positions_rad=tensor(q))
    assert (legacy.body_twist_world[:, :3]-derivative).norm() > .01


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_articulated_velocity_compiled_execution_updates_encoder_rates(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA graphs")
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_representative_order8_morphology(physical)
    eager = BatchedRigidBodyControlModelBuilder(morphology, physical)
    captured = BatchedRigidBodyControlModelBuilder(morphology, physical)
    captured._use_cuda_graph = True
    captured._use_cpu_compile = device == "cpu"
    pose = torch.zeros(1, eager.module_count, 7, device=device, dtype=torch.float64)
    pose[..., 6] = 1.
    twist = torch.zeros(1, eager.module_count, 6, device=device, dtype=torch.float64)
    twist[:, -1, 1] = .15
    q = torch.ones(1, eager.module_count, eager.local_joint_count, device=device, dtype=torch.float64)*.1
    with torch.no_grad():
        for rate in [.0, .2, -.3]:
            inputs = dict(module_pose_world=pose, module_twist_world=twist,
                local_joint_positions_rad=q, local_joint_velocities_rad=torch.full_like(q, rate))
            actual, expected = captured.build(**inputs), eager.build(**inputs)
            torch.testing.assert_close(actual.body_twist_world, expected.body_twist_world, rtol=1e-12, atol=1e-12)


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


def test_internal_joint_gravity_matches_complete_robot_potential_derivative():
    from amsrr.controllers.articulated_joint_load import ArticulatedJointLoadModel
    from amsrr.robot_model.whole_structure_kinematics import WholeStructureKinematics, ordered_global_dock_joint_ids
    physical = build_physical_model_from_config('configs/robot/robot_model.yaml')
    morphology = build_representative_order8_morphology(physical)
    reference = BatchedRigidBodyControlModelBuilder(morphology, physical)
    constrained = BatchedRigidBodyControlModelBuilder(morphology, physical)
    constrained._internal_load_model = ArticulatedJointLoadModel(morphology, physical)
    ids = ordered_global_dock_joint_ids(morphology, physical)
    kin = WholeStructureKinematics()
    base = (0., 0., 1., .1, .2, .3, math.sqrt(.86))
    q = {j: .1 * math.sin(i+1) for i, j in enumerate(ids)}
    def inputs(positions):
        fk = kin.forward(morphology, physical, positions, base, ())
        poses = torch.tensor([[fk.module_root_poses_world[m] for m in reference.module_ids]], dtype=torch.float64)
        local = torch.zeros(1, reference.module_count, reference.local_joint_count, dtype=torch.float64)
        for j, value in positions.items():
            module, name = j.split(':', 1)
            local[0, reference.module_ids.index(int(module.removeprefix('module_'))), reference.local_joint_ids.index(name)] = value
        return dict(module_pose_world=poses, module_twist_world=torch.zeros(1, reference.module_count, 6, dtype=torch.float64), local_joint_positions_rad=local)
    with torch.no_grad():
        actual = constrained.build(**inputs(q))
        derivatives = []
        epsilon = 1e-5
        for j in constrained._internal_load_model.joint_ids:
            plus, minus = dict(q), dict(q)
            plus[j] += epsilon; minus[j] -= epsilon
            a, b = reference.build(**inputs(plus)), reference.build(**inputs(minus))
            derivatives.append(float(-9.81 * a.total_mass_kg * (a.body_pose_world[0, 2]-b.body_pose_world[0, 2])/(2*epsilon)))
    torch.testing.assert_close(actual.joint_gravity_load_nm[0], torch.tensor(derivatives, dtype=torch.float64), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_internal_load_compiled_model_refreshes_joints_and_contact_jacobian(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from amsrr.controllers.articulated_joint_load import ArticulatedJointLoadModel
    from amsrr.feasibility.articulated_reachability import resolve_mesh_backed_anchor_references
    physical = build_physical_model_from_config('configs/robot/robot_model.yaml')
    morphology = build_representative_order8_morphology(physical)
    refs = resolve_mesh_backed_anchor_references(morphology, physical, [a.anchor_id for a in morphology.robot_anchors[:2]])
    eager = BatchedRigidBodyControlModelBuilder(morphology, physical)
    captured = BatchedRigidBodyControlModelBuilder(morphology, physical)
    for builder in (eager, captured):
        builder._internal_load_model = ArticulatedJointLoadModel(morphology, physical, refs)
    captured._use_cuda_graph = True
    captured._use_cpu_compile = device == 'cpu'
    pose = torch.zeros(1, eager.module_count, 7, device=device, dtype=torch.float64); pose[..., 6] = 1.
    twist = torch.zeros(1, eager.module_count, 6, device=device, dtype=torch.float64)
    q = torch.zeros(1, eager.module_count, eager.local_joint_count, device=device, dtype=torch.float64)
    with torch.no_grad():
        for angle in [.1, -.2]:
            q.fill_(angle)
            inputs = dict(module_pose_world=pose, module_twist_world=twist, local_joint_positions_rad=q)
            expected, actual = eager.build(**inputs), captured.build(**inputs)
            for name in ['joint_load_matrix', 'joint_gravity_load_nm', 'joint_mass_jacobian', 'joint_contact_jacobian', 'joint_angular_mass_matrix', 'joint_centrifugal_load_nm']:
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=1e-10, atol=1e-10)


def test_vectoring_rate_prediction_matches_authored_and_readback_drive_limit():
    from amsrr.controllers.qpid_controller import _joint_velocity_limits
    from amsrr.simulation.order9_actuator_runtime import order9_actuator_runtime_values
    physical = build_physical_model_from_config('configs/robot/robot_model.yaml')
    morphology = build_representative_order8_morphology(physical)
    builder = BatchedRigidBodyControlModelBuilder(morphology, physical)
    pose = torch.zeros(1, builder.module_count, 7, dtype=torch.float64); pose[..., 6] = 1.
    model = builder.build(module_pose_world=pose,
        module_twist_world=torch.zeros(1, builder.module_count, 6, dtype=torch.float64),
        local_joint_positions_rad=torch.zeros(1, builder.module_count, builder.local_joint_count, dtype=torch.float64))
    authored = order9_actuator_runtime_values(physical).gimbal_velocity_limit
    assert authored == pytest.approx(3.)
    assert next(j for j in physical.joints if j.joint_id == 'gimbal1').velocity_limit > authored
    torch.testing.assert_close(model.vectoring_velocity_limit_radps,
        torch.full_like(model.vectoring_velocity_limit_radps, authored))
    assert _joint_velocity_limits(physical)['gimbal1'] == (-authored, authored)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compiled_dynamics_invalidates_when_internal_load_model_changes(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from amsrr.controllers.articulated_joint_load import ArticulatedJointLoadModel
    physical = build_physical_model_from_config('configs/robot/robot_model.yaml')
    morphology = build_representative_order8_morphology(physical)
    builder = BatchedRigidBodyControlModelBuilder(morphology, physical)
    builder._use_cuda_graph = True
    builder._use_cpu_compile = device == 'cpu'
    pose = torch.zeros(1, builder.module_count, 7, device=device, dtype=torch.float64)
    pose[..., 6] = 1.
    inputs = dict(module_pose_world=pose,
        module_twist_world=torch.zeros(1, builder.module_count, 6, device=device, dtype=torch.float64),
        local_joint_positions_rad=torch.full((1, builder.module_count, builder.local_joint_count), .1, device=device, dtype=torch.float64))
    with torch.no_grad():
        assert builder.build(**inputs).joint_load_matrix is None
        builder._internal_load_model = ArticulatedJointLoadModel(morphology, physical)
        actual = builder.build(**inputs)
        reference = BatchedRigidBodyControlModelBuilder(morphology, physical)
        reference._internal_load_model = builder._internal_load_model
        torch.testing.assert_close(actual.joint_load_matrix, reference.build(**inputs).joint_load_matrix, atol=1e-10, rtol=1e-10)
        builder._internal_load_model = None
        assert builder.build(**inputs).joint_load_matrix is None


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_post_step_kinematics_matches_full_model_without_evaluating_loads(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    physical = build_physical_model_from_config('configs/robot/robot_model.yaml')
    morphology = build_representative_order8_morphology(physical)
    builder = BatchedRigidBodyControlModelBuilder(morphology, physical)
    builder._use_cuda_graph = device == 'cuda'
    class UnusedLoads:
        def evaluate(self, **kwargs):
            raise AssertionError('post-step observation must not construct allocation loads')
    previous = None
    with torch.no_grad():
        for batch, shift in [(1, .1), (1, -.3), (3, .2)]:
            pose = torch.zeros(batch, builder.module_count, 7, device=device)
            pose[..., 0] = torch.arange(builder.module_count, device=device) * .3
            pose[..., 5], pose[..., 6] = math.sin(shift), math.cos(shift)
            twist = torch.full((*pose.shape[:2], 6), shift, device=device)
            q = torch.full((*pose.shape[:2], builder.local_joint_count), shift, device=device)
            inputs = dict(module_pose_world=pose, module_twist_world=twist,
                local_joint_positions_rad=q, local_joint_velocities_rad=q*.7)
            builder._internal_load_model = None
            expected = builder.build(**inputs)
            builder._internal_load_model = UnusedLoads()
            actual = builder.build_kinematics(**inputs)
            for field in fields(actual):
                torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0)
            if previous:
                old, values = previous
                for name, value in values.items():
                    torch.testing.assert_close(getattr(old, name), value, rtol=0, atol=0)
            previous = actual, {f.name: getattr(actual, f.name).clone() for f in fields(actual)}
        inputs['local_joint_velocities_rad'][0, 0, 0] = float('nan')
        with pytest.raises(ValueError, match='finite'):
            builder.build_kinematics(**inputs)


def test_post_step_model_reuse_requires_identical_unmodified_owned_observations():
    physical=build_physical_model_from_config('configs/robot/robot_model.yaml')
    morphology=build_representative_order8_morphology(physical)
    b=BatchedRigidBodyControlModelBuilder(morphology,physical)
    b._reuse_post_step_model=True
    pose=torch.zeros(2,b.module_count,7);pose[...,6]=1.
    twist=torch.zeros(2,b.module_count,6)
    q=torch.zeros(2,b.module_count,b.local_joint_count)
    inputs=dict(module_pose_world=pose,module_twist_world=twist,
                local_joint_positions_rad=q,local_joint_velocities_rad=torch.zeros_like(q))
    calls=[];original=b._build
    def counted(*a,**kw):
        calls.append(1);return original(*a,**kw)
    b._build=counted
    post=b.build_kinematics(**inputs)
    full=b.build(**inputs)
    assert len(calls)==1
    torch.testing.assert_close(post.body_pose_world,full.body_pose_world,rtol=0,atol=0)
    q.add_(.03)
    changed=b.build(**inputs)
    assert len(calls)==2
    assert not torch.equal(changed.inertia_body,full.inertia_body)
    b.build_kinematics(**inputs)
    before=len(calls)
    b.build(**{**inputs,'module_pose_world':pose.clone()})
    assert len(calls)==before+1
