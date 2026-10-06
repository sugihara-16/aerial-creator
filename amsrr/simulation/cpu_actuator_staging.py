"""Copy implicit-actuator outputs directly between Isaac's CPU buffers.

Actuator.compute still owns the PD calculation and clipping. This only replaces
the two Warp scatter launches per group with indexed copies of the same fields.
The adapter is bound to one articulation and never changes Isaac's global class.
"""

from types import MethodType

import torch


class CpuActuatorStaging:
    def __init__(self, action_type, to_tensor):
        self.action_type = action_type
        self.to_tensor = to_tensor
        self.cache = None

    def apply(self, robot):
        data = robot._data
        buffers = (
            robot._joint_pos_target_sim,
            robot._joint_vel_target_sim,
            robot._joint_effort_target_sim,
            data.computed_torque,
            data.applied_torque,
            data.gear_ratio,
            data.soft_joint_vel_limits,
        )
        # A reinitialized articulation must not write through old tensor views.
        key = tuple(id(value) for value in buffers)
        if self.cache is None or self.cache[0] != key:
            self.cache = (key, tuple(self.to_tensor(value) for value in buffers))
        outputs = self.cache[1]
        for actuator in robot.actuators.values():
            indices = actuator.joint_indices
            selection = slice(None) if indices is None else indices
            action = self.action_type(
                joint_positions=data.joint_pos_target.torch[:, selection],
                joint_velocities=data.joint_vel_target.torch[:, selection],
                joint_efforts=data.joint_effort_target.torch[:, selection],
                joint_indices=indices,
            )
            action = actuator.compute(
                action,
                joint_pos=data.joint_pos.torch[:, selection],
                joint_vel=data.joint_vel.torch[:, selection],
            )
            values = (
                action.joint_positions,
                action.joint_velocities,
                action.joint_efforts,
                actuator.computed_effort,
                actuator.applied_effort,
                getattr(actuator, "gear_ratio", None),
                actuator.velocity_limit,
            )
            for destination, value in zip(outputs, values, strict=True):
                if value is None:
                    continue
                if isinstance(selection, slice):
                    destination[:, selection].copy_(value)
                else:
                    destination.index_copy_(1, torch.as_tensor(selection).long(), value)


class CpuJointTargetStaging:
    """Equivalent scatter for whole-batch, partial-joint CPU target writes."""

    def __init__(self, original, field):
        self.original = original
        self.field = field
        self.indices = None

    def write(self, robot, *, target, joint_ids=None, env_ids=None, full_data=False):
        # Retain Isaac's general path for subsets, Warp inputs and full layouts.
        if (
            env_ids is not None
            or full_data
            or not isinstance(target, torch.Tensor)
            or target.device.type != "cpu"
            or target.dtype != torch.float32
            or target.requires_grad
            or not isinstance(joint_ids, torch.Tensor)
            or joint_ids.device.type != "cpu"
            or joint_ids.ndim != 1
            or joint_ids.dtype not in (torch.int32, torch.int64)
        ):
            return self.original(
                target=target, joint_ids=joint_ids, env_ids=env_ids, full_data=full_data
            )
        destination = getattr(robot.data, self.field).torch
        if target.shape != (destination.shape[0], joint_ids.numel()):
            raise ValueError("CPU joint target shape differs from selected joints")
        cached = self.indices
        version = None if torch.is_inference(joint_ids) else joint_ids._version
        if (
            cached is None
            or cached[0] is not joint_ids
            or cached[1] != version
            or cached[2] != destination.shape[1]
            or (version is None and not torch.equal(joint_ids, cached[3]))
        ):
            indices = joint_ids.to(dtype=torch.int64, copy=True)
            if (
                indices.numel() == 0
                or int(indices.min()) < 0
                or int(indices.max()) >= destination.shape[1]
                or indices.unique().numel() != indices.numel()
            ):
                return self.original(
                    target=target,
                    joint_ids=joint_ids,
                    env_ids=env_ids,
                    full_data=full_data,
                )
            cached = (joint_ids, version, destination.shape[1], indices)
            self.indices = cached
        destination.index_copy_(1, cached[3], target)


class CpuSelectedWrenchStaging:
    """Send the same local wrench through a view of the selected rigid bodies."""

    def __init__(
        self,
        original,
        view,
        selected,
        body_count,
        environment_count,
        to_tensor,
        to_array,
        view_indices,
    ):
        self.original, self.view = original, view
        self.selected = torch.tensor(selected, dtype=torch.long)
        self.other = torch.ones(body_count, dtype=torch.bool)
        self.other[self.selected] = False
        self.body_count, self.environment_count = body_count, environment_count
        self.to_tensor, self.to_array, self.view_indices = (
            to_tensor,
            to_array,
            view_indices,
        )

    def apply(self, force_data, torque_data, position_data, indices, is_global):
        def general():
            return self.original(
                force_data, torque_data, position_data, indices, is_global
            )

        if (
            position_data is not None
            or is_global
            or force_data is None
            or torque_data is None
        ):
            return general()
        force, torque, environments = map(
            self.to_tensor, (force_data, torque_data, indices)
        )
        if (
            force.device.type != "cpu"
            or torque.device.type != "cpu"
            or environments.device.type != "cpu"
            or force.dtype != torch.float32
            or torque.dtype != torch.float32
            or environments.dtype not in (torch.int32, torch.uint32)
            or not force.is_contiguous()
            or not torque.is_contiguous()
            or force.numel() != self.body_count * 3
            or torque.numel() != force.numel()
            or not torch.equal(
                environments.to(dtype=torch.int64),
                torch.arange(self.environment_count),
            )
        ):
            return general()
        force, torque = force.reshape(-1, 3), torque.reshape(-1, 3)
        # Any other caller's force, including nonfinite input, retains the full
        # standard path. No nonzero wrench is silently omitted.
        if bool(torch.count_nonzero(force[self.other])) or bool(
            torch.count_nonzero(torque[self.other])
        ):
            return general()
        return self.view.apply_forces_and_torques_at_position(
            self.to_array(force.index_select(0, self.selected)),
            self.to_array(torque.index_select(0, self.selected)),
            None,
            self.view_indices,
            False,
        )


def install_cpu_selected_wrench_staging(robot, body_indices):
    if str(robot.device) != "cpu" or hasattr(robot, "_request_selected_wrench_staging"):
        return
    root = getattr(robot, "root_view", None)
    original = getattr(root, "apply_forces_and_torques_at_position", None)
    function = getattr(original, "__func__", None)
    if (
        getattr(function, "__module__", None) != "omni.physics.tensors.api"
        or getattr(function, "__qualname__", None)
        != "ArticulationView.apply_forces_and_torques_at_position"
    ):
        return
    from omni.physics.tensors.frontend_warp import FrontendWarp
    import warp as wp

    if type(root._frontend) is not FrontendWarp:
        return
    links = root.link_paths
    if (
        not links
        or not body_indices
        or len(set(body_indices)) != len(body_indices)
        or any(
            not isinstance(i, int)
            or isinstance(i, bool)
            or i < 0
            or any(i >= len(row) for row in links)
            for i in body_indices
        )
    ):
        raise ValueError("invalid selected wrench body indices")
    all_paths = [path for row in links for path in row]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("articulation body paths are not unique")
    selected_paths = [row[i] for row in links for i in body_indices]
    view = robot._physics_sim_view.create_rigid_body_view(selected_paths)
    paths = view.prim_paths
    if len(paths) != len(selected_paths) or set(paths) != set(selected_paths):
        raise ValueError("selected wrench view body paths differ")
    lookup = {path: i for i, path in enumerate(all_paths)}
    writer = CpuSelectedWrenchStaging(
        original,
        view,
        [lookup[p] for p in paths],
        len(all_paths),
        len(links),
        wp.to_torch,
        wp.from_torch,
        wp.array(list(range(len(paths))), dtype=wp.int32, device="cpu"),
    )
    robot._request_selected_wrench_staging = writer

    def apply(asset, *args, **kwargs):
        return writer.apply(*args, **kwargs)

    root.apply_forces_and_torques_at_position = MethodType(apply, root)


def install_cpu_actuator_staging(robot):
    """Use the host copy path only for the standard implicit actuator contract."""
    if str(robot.device) != "cpu" or hasattr(robot, "_request_cpu_actuator_staging"):
        return
    method = getattr(robot, "_apply_actuator_model", None)
    function = getattr(method, "__func__", None)
    if (
        getattr(function, "__module__", None)
        != "isaaclab_physx.assets.articulation.articulation"
        or getattr(function, "__qualname__", None)
        != "Articulation._apply_actuator_model"
        or not all(
            hasattr(robot, name)
            for name in (
                "_joint_pos_target_sim",
                "_joint_vel_target_sim",
                "_joint_effort_target_sim",
            )
        )
    ):
        return
    from isaaclab.actuators import ImplicitActuator
    from isaaclab.utils.types import ArticulationActions
    import warp as wp

    if getattr(robot, "_has_newton_actuators", False) or not all(
        type(a) is ImplicitActuator and a.joint_indices is not None
        for a in robot.actuators.values()
    ):
        return

    def to_tensor(value):
        return value.torch if hasattr(value, "torch") else wp.to_torch(value)

    staging = CpuActuatorStaging(ArticulationActions, to_tensor)
    robot._request_cpu_actuator_staging = staging
    robot._apply_actuator_model = MethodType(lambda asset: staging.apply(asset), robot)
    for kind, field in (
        ("position", "joint_pos_target"),
        ("velocity", "joint_vel_target"),
        ("effort", "joint_effort_target"),
    ):
        name = f"set_joint_{kind}_target_index"
        original = getattr(robot, name)
        function = getattr(original, "__func__", None)
        if (
            getattr(function, "__module__", None)
            != "isaaclab_physx.assets.articulation.articulation"
            or getattr(function, "__qualname__", None) != f"Articulation.{name}"
        ):
            continue
        writer = CpuJointTargetStaging(original, field)

        def write(asset, *, _writer=writer, **kwargs):
            return _writer.write(asset, **kwargs)

        setattr(robot, name, MethodType(write, robot))
