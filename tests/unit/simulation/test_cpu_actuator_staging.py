from types import SimpleNamespace

import pytest
import torch

from amsrr.simulation.cpu_actuator_staging import CpuActuatorStaging


def test_selected_wrench_writer_preserves_path_order_and_external_wrenches():
    from amsrr.simulation.cpu_actuator_staging import CpuSelectedWrenchStaging

    captured, general = [], []
    view = SimpleNamespace(
        apply_forces_and_torques_at_position=lambda *a: captured.append(a)
    )
    selected = [
        5,
        1,
        4,
        0,
    ]  # Rigid-body view ordering differs from articulation ordering.
    writer = CpuSelectedWrenchStaging(
        lambda *a: general.append(a),
        view,
        selected,
        6,
        2,
        lambda x: x,
        lambda x: x,
        torch.arange(4),
    )
    force, torque = torch.zeros(6, 3), torch.zeros(6, 3)
    force[selected] = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    torque[selected] = -force[selected]
    args = [force, torque, None, torch.arange(2, dtype=torch.int32), False]
    writer.apply(*args)
    assert not general and len(captured) == 1
    assert torch.equal(captured[0][0], force[selected])
    assert torch.equal(captured[0][1], torque[selected])
    force[3, 0] = 0.001  # A nonrotor disturbance must never be discarded.
    writer.apply(*args)
    assert len(general) == 1 and len(captured) == 1
    force[3, 0] = float("nan")
    writer.apply(*args)
    assert len(general) == 2
    force[3, 0] = 0
    writer.apply(force, torque, None, torch.tensor([1, 0]), False)
    writer.apply(force, torque, None, args[3], True)
    writer.apply(force, torque, torch.zeros_like(force), args[3], False)
    assert len(general) == 5 and len(captured) == 1
    writer.apply(force, torque, None, torch.tensor([0., 1.]), False)
    assert len(general) == 6 and len(captured) == 1
    force[5] = 9
    writer.apply(*args)
    assert torch.equal(captured[-1][0][0], torch.full((3,), 9.0))


def proxy(tensor):
    return SimpleNamespace(torch=tensor)


@pytest.mark.parametrize(
    "selection", [slice(None), slice(1, 6, 2), torch.tensor([4, 0, 2])]
)
def test_cpu_staging_matches_per_joint_copy_and_refreshes_inputs(selection):
    torch.manual_seed(42)
    n, joints = 3, 6
    ids = torch.arange(joints)[selection]
    a = SimpleNamespace(
        joint_indices=selection,
        velocity_limit=torch.rand(n, len(ids)),
        gear_ratio=torch.rand(n, len(ids)),
    )

    def compute(action, *, joint_pos, joint_vel):
        a.computed_effort = (
            3.0 * (action.joint_positions - joint_pos)
            + 0.4 * (action.joint_velocities - joint_vel)
            + action.joint_efforts
        )
        a.applied_effort = a.computed_effort.clamp(-2.0, 2.0)
        return action

    a.compute = compute
    data = SimpleNamespace(
        **{
            name: proxy(torch.randn(n, joints))
            for name in (
                "joint_pos_target",
                "joint_vel_target",
                "joint_effort_target",
                "joint_pos",
                "joint_vel",
                "computed_torque",
                "applied_torque",
                "gear_ratio",
                "soft_joint_vel_limits",
            )
        }
    )
    robot = SimpleNamespace(
        _data=data,
        actuators={"motors": a},
        _joint_pos_target_sim=torch.randn(n, joints),
        _joint_vel_target_sim=torch.randn(n, joints),
        _joint_effort_target_sim=torch.randn(n, joints),
    )
    adapter = CpuActuatorStaging(
        SimpleNamespace, lambda v: v.torch if hasattr(v, "torch") else v
    )
    for step in range(3):
        data.joint_pos_target.torch.add_(0.12)
        data.joint_vel.torch.sub_(0.04)
        # A subsequent scene initialization can replace buffers on the same asset.
        if step == 2:
            robot._joint_pos_target_sim = torch.randn(n, joints)
            data.applied_torque = proxy(torch.randn(n, joints))
        outputs = [
            robot._joint_pos_target_sim,
            robot._joint_vel_target_sim,
            robot._joint_effort_target_sim,
            data.computed_torque.torch,
            data.applied_torque.torch,
            data.gear_ratio.torch,
            data.soft_joint_vel_limits.torch,
        ]
        expected = [x.clone() for x in outputs]
        q, v, e = [
            getattr(data, key).torch[:, ids]
            for key in ("joint_pos_target", "joint_vel_target", "joint_effort_target")
        ]
        effort = (
            3.0 * (q - data.joint_pos.torch[:, ids])
            + 0.4 * (v - data.joint_vel.torch[:, ids])
            + e
        )
        values = [
            q,
            v,
            e,
            effort,
            effort.clamp(-2.0, 2.0),
            a.gear_ratio,
            a.velocity_limit,
        ]
        for dst, src in zip(expected, values):
            for col, joint in enumerate(ids.tolist()):
                dst[:, joint] = src[:, col]
        adapter.apply(robot)
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(a.computed_effort, effort, rtol=0, atol=0)
        torch.testing.assert_close(
            a.applied_effort, effort.clamp(-2.0, 2.0), rtol=0, atol=0
        )


def test_absent_target_and_gear_output_preserve_existing_buffers():
    tensor = lambda: torch.randn(2, 3)
    data = SimpleNamespace(
        **{
            name: proxy(tensor())
            for name in (
                "joint_pos_target",
                "joint_vel_target",
                "joint_effort_target",
                "joint_pos",
                "joint_vel",
                "computed_torque",
                "applied_torque",
                "gear_ratio",
                "soft_joint_vel_limits",
            )
        }
    )
    actuator = SimpleNamespace(
        joint_indices=slice(None),
        computed_effort=tensor(),
        applied_effort=tensor(),
        velocity_limit=tensor(),
        compute=lambda action, **state: SimpleNamespace(
            joint_positions=None, joint_velocities=None, joint_efforts=None
        ),
    )
    robot = SimpleNamespace(
        _data=data,
        actuators={"a": actuator},
        _joint_pos_target_sim=tensor(),
        _joint_vel_target_sim=tensor(),
        _joint_effort_target_sim=tensor(),
    )
    kept = [
        robot._joint_pos_target_sim,
        robot._joint_vel_target_sim,
        robot._joint_effort_target_sim,
        data.gear_ratio.torch,
    ]
    reference = [x.clone() for x in kept]
    CpuActuatorStaging(
        SimpleNamespace, lambda v: v.torch if hasattr(v, "torch") else v
    ).apply(robot)
    for a, b in zip(kept, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_install_leaves_unsupported_backend_unchanged(device):
    from amsrr.simulation.cpu_actuator_staging import install_cpu_actuator_staging

    original = lambda: None
    robot = SimpleNamespace(device=device, _apply_actuator_model=original)
    install_cpu_actuator_staging(robot)
    assert robot._apply_actuator_model is original
    assert not hasattr(robot, "_request_cpu_actuator_staging")


@pytest.mark.parametrize("inference", [False, True])
def test_joint_target_scatter_preserves_other_joints_and_tracks_index_changes(
    inference,
):
    from amsrr.simulation.cpu_actuator_staging import CpuJointTargetStaging

    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos_target=proxy(torch.arange(18).reshape(3, 6).float())
        )
    )

    def unexpected(**kw):
        raise AssertionError("unexpected general writer")

    writer = CpuJointTargetStaging(unexpected, "joint_pos_target")
    with torch.inference_mode(inference):
        ids = torch.tensor([4, 1], dtype=torch.int32)
        for step in range(3):
            if step == 1:
                ids[0] = 3
            if step == 2:
                robot.data.joint_pos_target = proxy(torch.zeros(3, 6))
            targets = torch.randn(3, 2)
            expected = robot.data.joint_pos_target.torch.clone()
            for j, col in enumerate(ids.tolist()):
                expected[:, col] = targets[:, j]
            writer.write(robot, target=targets, joint_ids=ids)
            assert torch.equal(robot.data.joint_pos_target.torch, expected)


def test_joint_target_staging_preserves_general_writer_and_rejects_wrong_shape():
    from amsrr.simulation.cpu_actuator_staging import CpuJointTargetStaging

    calls = []
    robot = SimpleNamespace(
        data=SimpleNamespace(joint_pos_target=proxy(torch.zeros(2, 4)))
    )
    writer = CpuJointTargetStaging(lambda **kw: calls.append(kw), "joint_pos_target")
    ids = torch.tensor([0, 2], dtype=torch.int32)
    value = torch.ones(2, 2)
    writer.write(robot, target=value, joint_ids=ids, env_ids=[0])
    writer.write(robot, target=value, joint_ids=ids, full_data=True)
    writer.write(robot, target=value, joint_ids=torch.tensor([0, 0]))
    assert len(calls) == 3
    assert not robot.data.joint_pos_target.torch.any()
    with pytest.raises(ValueError, match="shape"):
        writer.write(robot, target=torch.ones(2, 3), joint_ids=ids)
