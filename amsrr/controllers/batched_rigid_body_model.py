from __future__ import annotations

"""Tensorized quasi-static rigid-body model for one topology bucket.

This is the batched equivalent of :mod:`amsrr.controllers.rigid_body_model`.
It recomputes link kinematics, aggregate CoM/inertia, rotor origins/axes and
the virtual-thrust allocation columns from current joint positions every
control cycle.  Python loops describe the fixed link tree; all environment and
module state remains in torch tensors on the rollout device.
"""

from dataclasses import dataclass, fields

import torch

from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import JointModel, PhysicalModel
from amsrr.controllers.rigid_body_model import _joint_limits


@dataclass(frozen=True)
class BatchedRigidBodyControlModel:
    body_pose_world: torch.Tensor
    body_twist_world: torch.Tensor
    total_mass_kg: torch.Tensor
    inertia_body_matrix: torch.Tensor
    inertia_body: torch.Tensor
    virtual_x_wrench_columns: torch.Tensor
    virtual_z_wrench_columns: torch.Tensor
    current_vectoring_angles_rad: torch.Tensor
    thrust_min_n: torch.Tensor
    thrust_max_n: torch.Tensor
    vectoring_lower_rad: torch.Tensor
    vectoring_upper_rad: torch.Tensor
    vectoring_velocity_limit_radps: torch.Tensor
    rotor_module_ids: tuple[int, ...]
    rotor_local_ids: tuple[str, ...]
    vectoring_global_joint_ids: tuple[str, ...]
    joint_load_matrix: torch.Tensor | None = None
    joint_gravity_load_nm: torch.Tensor | None = None
    joint_mass_jacobian: torch.Tensor | None = None
    joint_contact_jacobian: torch.Tensor | None = None
    joint_angular_mass_matrix: torch.Tensor | None = None
    joint_centrifugal_load_nm: torch.Tensor | None = None


@dataclass(frozen=True)
class BatchedCentroidalKinematics:
    """Post-physics observation fields; no allocation or joint-load matrices."""
    body_pose_world: torch.Tensor
    body_twist_world: torch.Tensor
    current_vectoring_angles_rad: torch.Tensor


@dataclass(frozen=True)
class _TensorTransform:
    rotation: torch.Tensor
    translation: torch.Tensor


class BatchedRigidBodyControlModelBuilder:
    """Build batched controller models for a fixed morphology topology."""

    def __init__(
        self, morphology_graph: MorphologyGraph, physical_model: PhysicalModel
    ) -> None:
        morphology_graph.validate()
        physical_model.validate()
        self.morphology_graph = MorphologyGraph.from_dict(
            morphology_graph.to_dict()
        )
        self.physical_model = PhysicalModel.from_dict(physical_model.to_dict())
        # Opted into only by the nominal request executor. Historical callers
        # retain the eager path, including autograd and arbitrary batch sizes.
        self._use_cuda_graph = False
        self._use_cpu_compile = False
        self._use_cpu_native = False
        self._cpu_native_model = None
        self._reuse_post_step_model = False
        self._post_step_cache = None
        self._cpu_compiled_build = None
        self._cpu_compiled_kinematics = None
        self._cuda_graph_cache = None
        self._kinematics_graph_cache = None
        self._constant_cache = {}
        self._internal_load_model = None
        self.module_ids = tuple(
            sorted(module.module_id for module in morphology_graph.modules)
        )
        if morphology_graph.base_module_id not in self.module_ids:
            raise ValueError("batched rigid-body model has no base module")
        self.base_module_index = self.module_ids.index(
            morphology_graph.base_module_id
        )
        self.local_joint_ids = tuple(
            joint.joint_id for joint in physical_model.joints
        )
        self._joint_index = {
            joint_id: index for index, joint_id in enumerate(self.local_joint_ids)
        }
        self.link_ids = tuple(link.link_id for link in physical_model.links)
        self._links_by_id = {link.link_id: link for link in physical_model.links}
        self._joints_by_id = {
            joint.joint_id: joint for joint in physical_model.joints
        }
        self._joints_by_parent: dict[str, list[JointModel]] = {}
        child_links: set[str] = set()
        for joint in physical_model.joints:
            self._joints_by_parent.setdefault(joint.parent_link, []).append(joint)
            child_links.add(joint.child_link)
        self._roots = tuple(sorted(set(self.link_ids) - child_links))
        if not self._roots:
            raise ValueError("batched rigid-body physical model has no root link")
        self._base_link = _module_base_link(physical_model, set(self.link_ids))
        parent_joint = {j.child_link: j for j in physical_model.joints}
        def ancestors(link):
            result = set()
            while link in parent_joint:
                joint = parent_joint[link]
                result.add(joint.joint_id)
                link = joint.parent_link
            return result
        base_ancestors = ancestors(self._base_link)
        link_ancestors = {link: ancestors(link) for link in self.link_ids}
        # Relative module-frame COM Jacobian. A joint upstream of the module
        # frame moves that frame too, hence the signed ancestor difference.
        self._joint_com_mass_weights = {
            j.joint_id: [self._links_by_id[link].mass_kg *
                (int(j.joint_id in link_ancestors[link]) - int(j.joint_id in base_ancestors))
                for link in self.link_ids]
            for j in physical_model.joints if j.joint_type != "fixed"
        }
        self._rotor_specs = tuple(
            (module_id, rotor)
            for module_id in self.module_ids
            for rotor in sorted(physical_model.rotors, key=lambda item: item.rotor_id)
        )
        for _, rotor in self._rotor_specs:
            if len(rotor.vectoring_joint_ids) != 1:
                raise ValueError(
                    "Order9 batched allocator requires one-axis vectoring rotors"
                )

    @property
    def module_count(self) -> int:
        return len(self.module_ids)

    @property
    def local_joint_count(self) -> int:
        return len(self.local_joint_ids)

    @property
    def rotor_count(self) -> int:
        return len(self._rotor_specs)

    def build(
        self,
        *,
        module_pose_world: torch.Tensor,
        module_twist_world: torch.Tensor,
        local_joint_positions_rad: torch.Tensor,
        local_joint_velocities_rad: torch.Tensor | None = None,
    ) -> BatchedRigidBodyControlModel:
        inputs = self._validated_inputs(module_pose_world, module_twist_world,
            local_joint_positions_rad, local_joint_velocities_rad)
        cached = self._post_step_cache if self._reuse_post_step_model else None
        if (cached is not None and cached[2] is self._internal_load_model
                and (not torch.is_grad_enabled() or not any(x.requires_grad for x in inputs))
                and len(inputs) == len(cached[0])
                and all(value is previous and value._version == version
                        for value, (previous, version) in zip(inputs, cached[0]))):
            return cached[1]
        if (self._use_cpu_compile and module_pose_world.device.type == 'cpu'
                and (not torch.is_grad_enabled() or not any(x.requires_grad for x in inputs))):
            with torch.no_grad():
                return self._build_compiled_cpu(inputs)
        if (
            self._use_cuda_graph
            and module_pose_world.is_cuda
            and (not torch.is_grad_enabled() or not any(x.requires_grad for x in inputs))
        ):
            with torch.no_grad():
                return self._build_cuda_graph(inputs)
        return self._build(*inputs)

    def _validated_inputs(self, module_pose_world, module_twist_world,
                          local_joint_positions_rad, local_joint_velocities_rad):
        self._validate_inputs(
            module_pose_world, module_twist_world, local_joint_positions_rad
        )
        inputs = (module_pose_world, module_twist_world, local_joint_positions_rad)
        if local_joint_velocities_rad is not None:
            if (local_joint_velocities_rad.shape != local_joint_positions_rad.shape
                    or local_joint_velocities_rad.device != module_pose_world.device
                    or local_joint_velocities_rad.dtype != module_pose_world.dtype
                    or not bool(torch.isfinite(local_joint_velocities_rad).all())):
                raise ValueError("joint velocities must match finite position tensor")
            inputs = (*inputs, local_joint_velocities_rad)
        return inputs

    def build_kinematics(self, *, module_pose_world, module_twist_world,
                         local_joint_positions_rad, local_joint_velocities_rad=None):
        """Use identical COM/velocity math, without unused post-step dynamics."""
        inputs = self._validated_inputs(module_pose_world, module_twist_world,
            local_joint_positions_rad, local_joint_velocities_rad)
        if self._reuse_post_step_model and not any(value.requires_grad or value.is_inference() for value in inputs):
            # The next control call consumes these very same owned post-physics
            # observation tensors. Compute dynamics once, and retain only this
            # sample; any new tensor or in-place update invalidates the cache.
            model = self.build(module_pose_world=module_pose_world,
                module_twist_world=module_twist_world,
                local_joint_positions_rad=local_joint_positions_rad,
                local_joint_velocities_rad=local_joint_velocities_rad)
            self._post_step_cache = (tuple((value, value._version) for value in inputs), model, self._internal_load_model)
            return BatchedCentroidalKinematics(model.body_pose_world, model.body_twist_world,
                model.current_vectoring_angles_rad)
        if (self._use_cpu_compile and module_pose_world.device.type == 'cpu'
                and (not torch.is_grad_enabled() or not any(x.requires_grad for x in inputs))):
            with torch.no_grad():
                return self._build_compiled_cpu(inputs, kinematics_only=True)
        if (self._use_cuda_graph and module_pose_world.is_cuda
                and (not torch.is_grad_enabled() or not any(x.requires_grad for x in inputs))):
            with torch.no_grad():
                return self._build_cuda_graph(inputs, kinematics_only=True)
        return self._build(*inputs, kinematics_only=True)

    def _build_compiled_cpu(self, inputs, *, kinematics_only=False):
        """Compile the fixed topology for CPU, retaining fresh state every tick.

        Opt-in deployment path: compilation errors propagate; there is no
        silent fallback to a slower backend that could miss control deadlines.
        The nominal executor warms it before starting control.
        """
        if (self._use_cpu_native and self._internal_load_model is not None
                and len(inputs) == 4 and inputs[0].dtype in (torch.float32, torch.float64)):
            from amsrr.controllers.native_cpu_dynamics import NativeCpuControlModel
            key = (tuple((tuple(x.shape), x.dtype) for x in inputs), self._internal_load_model)
            if self._cpu_native_model is None or self._cpu_native_model[0] != key:
                self._cpu_native_model = (key, NativeCpuControlModel(self, inputs))
            result = self._cpu_native_model[1](*inputs)
            if kinematics_only:
                return BatchedCentroidalKinematics(result.body_pose_world,
                    result.body_twist_world, result.current_vectoring_angles_rad)
            return result
        key = (tuple((tuple(x.shape), x.dtype, x.device) for x in inputs), self._internal_load_model)
        cache_name = '_cpu_compiled_kinematics' if kinematics_only else '_cpu_compiled_build'
        cached = getattr(self, cache_name)
        if cached is None or cached[0] != key:
            reference = self._build(*inputs, kinematics_only=kinematics_only)
            result_type = type(reference)
            names = tuple(f.name for f in fields(reference) if isinstance(getattr(reference, f.name), torch.Tensor))
            metadata = {f.name: getattr(reference, f.name) for f in fields(reference) if f.name not in names}
            def compute(*args):
                result = self._build(*args, kinematics_only=kinematics_only)
                return tuple(getattr(result, name) for name in names)
            # CPU Inductor spends minutes compiling this unrolled link tree.
            # A static trace keeps cold-job setup bounded. Input validation
            # stays outside, shape/topology is in the key, and every pose/rate
            # remains a runtime tensor input.
            compiled = torch.jit.trace(compute, inputs, check_trace=False)
            cached = (key, compiled, names, metadata, result_type)
            setattr(self, cache_name, cached)
        _, compiled, names, metadata, result_type = cached
        return result_type(**dict(zip(names, compiled(*inputs))), **metadata)

    def _constant(self, values, *, device, dtype):
        if not self._use_cuda_graph:
            return torch.tensor(values, device=device, dtype=dtype)

        def freeze(value):
            if isinstance(value, (list, tuple)):
                return tuple(freeze(v) for v in value)
            return value

        key = (device, dtype, freeze(values))
        if key not in self._constant_cache:
            self._constant_cache[key] = torch.tensor(values, device=device, dtype=dtype)
        return self._constant_cache[key]

    def _build_cuda_graph(self, inputs, *, kinematics_only=False):
        key = (tuple((x.shape, x.device, x.dtype) for x in inputs), self._internal_load_model)
        cache_name = "_kinematics_graph_cache" if kinematics_only else "_cuda_graph_cache"
        cached = getattr(self, cache_name)
        if cached is None or cached[0] != key:
            # One graph per builder bounds memory when a caller changes shape.
            static = tuple(x.clone() for x in inputs)
            stream = torch.cuda.Stream(device=inputs[0].device)
            stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._build(*static, kinematics_only=kinematics_only)
            torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = self._build(*static, kinematics_only=kinematics_only)
            cached = (key, static, graph, result)
            setattr(self, cache_name, cached)
        _, static, graph, result = cached
        for destination, source in zip(static, inputs, strict=True):
            destination.copy_(source)
        graph.replay()
        # A post-step build must not overwrite the preceding controller result.
        return type(result)(**{
            field.name: value.clone() if isinstance(value, torch.Tensor) else value
            for field in fields(result)
            for value in (getattr(result, field.name),)
        })

    def _build(self, module_pose_world, module_twist_world, local_joint_positions_rad,
               local_joint_velocities_rad=None, *, kinematics_only=False):
        batch_size = module_pose_world.shape[0]
        device = module_pose_world.device
        dtype = module_pose_world.dtype
        module_rotation_world = _quaternion_to_matrix(module_pose_world[..., 3:7])
        module_translation_world = module_pose_world[..., :3]
        link_module, joint_axis_module = self._module_link_kinematics(
            local_joint_positions_rad
        )

        link_com_world: list[torch.Tensor] = []
        link_inertia_world: list[torch.Tensor] = []
        link_masses: list[float] = []
        world_link: dict[str, _TensorTransform] = {}
        for link_id in self.link_ids:
            local = link_module[link_id]
            rotation = module_rotation_world @ local.rotation
            translation = module_translation_world + (
                module_rotation_world @ local.translation.unsqueeze(-1)
            ).squeeze(-1)
            world_link[link_id] = _TensorTransform(rotation, translation)
            link = self._links_by_id[link_id]
            local_com = self._constant(
                link.local_com, device=device, dtype=dtype
            ).reshape(1, 1, 3, 1)
            com = translation + (rotation @ local_com).squeeze(-1)
            if not kinematics_only:
                inertia_local = self._constant(
                    _inertia6_to_matrix(link.inertia_kgm2),
                    device=device,
                    dtype=dtype,
                ).reshape(1, 1, 3, 3)
                inertia_world = rotation @ inertia_local @ rotation.transpose(-1, -2)
            link_com_world.append(com)
            if not kinematics_only:
                link_inertia_world.append(inertia_world)
            link_masses.append(float(link.mass_kg))

        masses = self._constant(link_masses, device=device, dtype=dtype)
        # [link, batch, module, xyz] -> [batch, module, link, xyz]
        com_by_link = torch.stack(link_com_world, dim=0).permute(1, 2, 0, 3)
        total_mass_scalar = masses.sum() * float(self.module_count)
        if self._use_cuda_graph:
            # Preserve the device/dtype reduction, without a host .item() that
            # would synchronize and cannot be recorded in a CUDA graph.
            total_mass = total_mass_scalar.expand(batch_size).clone()
        else:
            total_mass = torch.full(
                (batch_size,), float(total_mass_scalar.item()),
                device=device, dtype=dtype,
            )
        com_world = (
            com_by_link * masses.reshape(1, 1, -1, 1)
        ).sum(dim=(1, 2)) / total_mass.unsqueeze(-1)
        base_rotation_world = module_rotation_world[:, self.base_module_index]
        world_to_body = base_rotation_world.transpose(-1, -2)
        if not kinematics_only:
            inertia_by_link = torch.stack(link_inertia_world, dim=0).permute(
                1, 2, 0, 3, 4
            )
            relative_com_world = com_by_link - com_world[:, None, None, :]
            relative_com_body = (
                world_to_body[:, None, None]
                @ relative_com_world.unsqueeze(-1)
            ).squeeze(-1)
            inertia_at_link_body = (
                world_to_body[:, None, None]
                @ inertia_by_link
                @ base_rotation_world[:, None, None]
            )
            squared_norm = relative_com_body.square().sum(dim=-1)
            identity = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 1, 3, 3)
            parallel_axis = masses.reshape(1, 1, -1, 1, 1) * (
                squared_norm[..., None, None] * identity
                - relative_com_body.unsqueeze(-1)
                * relative_com_body.unsqueeze(-2)
            )
            inertia_body_matrix = (inertia_at_link_body + parallel_axis).sum(dim=(1, 2))

        base_pose = module_pose_world[:, self.base_module_index]
        body_pose = torch.cat((com_world, base_pose[:, 3:7]), dim=-1)
        base_twist = module_twist_world[:, self.base_module_index]
        base_origin = base_pose[:, :3]
        com_velocity = base_twist[:, :3] + torch.cross(
            base_twist[:, 3:6], com_world - base_origin, dim=-1
        )
        if local_joint_velocities_rad is not None:
            # Each module is observed independently. Dock articulation can move
            # the other modules substantially relative to the base module.
            velocity_by_link = module_twist_world[:, :, None, :3] + torch.cross(
                module_twist_world[:, :, None, 3:].expand_as(com_by_link),
                com_by_link - module_translation_world[:, :, None, :], dim=-1)
            momentum = (velocity_by_link * masses[None, None, :, None]).sum(dim=(1, 2))
            for joint_id, weights in self._joint_com_mass_weights.items():
                joint = self._joints_by_id[joint_id]
                signed_mass = self._constant(weights, device=device, dtype=dtype)
                axis = (module_rotation_world @ joint_axis_module[joint_id].unsqueeze(-1)).squeeze(-1)
                if joint.joint_type in {"revolute", "continuous"}:
                    parent = world_link[joint.parent_link]
                    origin = parent.translation + (parent.rotation @ self._constant(
                        joint.origin_xyz, device=device, dtype=dtype).reshape(1, 1, 3, 1)).squeeze(-1)
                    lever = ((com_by_link - origin[:, :, None, :]) * signed_mass[None, None, :, None]).sum(dim=2)
                    derivative = torch.cross(axis, lever, dim=-1)
                elif joint.joint_type == "prismatic":
                    derivative = axis * signed_mass.sum()
                else:
                    continue
                momentum = momentum + (derivative * local_joint_velocities_rad[:, :, self._joint_index[joint_id], None]).sum(dim=1)
            com_velocity = momentum / total_mass[:, None]
        body_twist = torch.cat((com_velocity, base_twist[:, 3:6]), dim=-1)

        if kinematics_only:
            angles = torch.stack([
                local_joint_positions_rad[:, self.module_ids.index(module_id),
                    self._joint_index[rotor.vectoring_joint_ids[0]]]
                for module_id, rotor in self._rotor_specs], dim=1)
            return BatchedCentroidalKinematics(body_pose, body_twist, angles)

        if device.type == 'cpu':
            # All rotor columns use the same algebra. Evaluate across the rotor
            # dimension instead of launching the same tiny operations per rotor.
            rotor_modules = [self.module_ids.index(m) for m, _ in self._rotor_specs]
            rotor_joints = [self._joints_by_id[r.vectoring_joint_ids[0]] for _, r in self._rotor_specs]
            rotor_positions = torch.stack([
                world_link[r.thrust_frame_link].translation[:, mi]
                for mi, (_, r) in zip(rotor_modules, self._rotor_specs)], dim=1)
            origin_body = (world_to_body[:, None] @
                (rotor_positions - com_world[:, None]).unsqueeze(-1)).squeeze(-1)
            arm_rotation = torch.stack([world_link[j.parent_link].rotation[:, mi]
                for mi, j in zip(rotor_modules, rotor_joints)], dim=1)
            local_z = self._constant([(0., 0., 1. if r.thrust_axis_local[2] >= 0. else -1.)
                for _, r in self._rotor_specs], device=device, dtype=dtype)
            z_world = (arm_rotation @ local_z[None, :, :, None]).squeeze(-1)
            z_axis_body = _normalize((world_to_body[:, None] @ z_world.unsqueeze(-1)).squeeze(-1))
            module_axes = torch.stack([joint_axis_module[j.joint_id][:, mi]
                for mi, j in zip(rotor_modules, rotor_joints)], dim=1)
            rotor_module_rotation = torch.stack([module_rotation_world[:, mi]
                for mi in rotor_modules], dim=1)
            axes_world = (rotor_module_rotation @ module_axes.unsqueeze(-1)).squeeze(-1)
            vectoring_axis_body = _normalize((world_to_body[:, None] @ axes_world.unsqueeze(-1)).squeeze(-1))
            x_axis_body = _normalize(torch.cross(vectoring_axis_body, z_axis_body, dim=-1))
            reactions = self._constant([float(r.reaction_torque_coeff_nm_per_n)
                for _, r in self._rotor_specs], device=device, dtype=dtype)[None, :, None]
            x_columns = _wrench_column(origin_body, x_axis_body, reactions)
            z_columns = _wrench_column(origin_body, z_axis_body, reactions)
            current_angles = torch.stack([local_joint_positions_rad[:, mi, self._joint_index[j.joint_id]]
                for mi, j in zip(rotor_modules, rotor_joints)], dim=1)
            thrust_min = [float(r.thrust_min_n) for _, r in self._rotor_specs]
            thrust_max = [float(r.thrust_max_n) for _, r in self._rotor_specs]
            angle_lower = [float(j.limit_lower or 0.) for j in rotor_joints]
            angle_upper = [float(j.limit_upper or 0.) for j in rotor_joints]
            angle_velocity = [float(_joint_limits(j, self.physical_model)["velocity"] or 0.) for j in rotor_joints]
            rotor_module_ids = [m for m, _ in self._rotor_specs]
            rotor_local_ids = [r.rotor_id for _, r in self._rotor_specs]
            vectoring_global_joint_ids = [f"module_{m}:{j.joint_id}" for (m, _), j in zip(self._rotor_specs, rotor_joints)]

            # Preserve the common result assembly and its owned tensor outputs.
            x_columns, z_columns = list(x_columns.unbind(1)), list(z_columns.unbind(1))
            current_angles = list(current_angles.unbind(1))
        else:
            x_columns: list[torch.Tensor] = []
            z_columns: list[torch.Tensor] = []
            current_angles: list[torch.Tensor] = []
            thrust_min: list[float] = []
            thrust_max: list[float] = []
            angle_lower: list[float] = []
            angle_upper: list[float] = []
            angle_velocity: list[float] = []
            rotor_module_ids: list[int] = []
            rotor_local_ids: list[str] = []
            vectoring_global_joint_ids: list[str] = []
            module_index_by_id = {
                module_id: index for index, module_id in enumerate(self.module_ids)
            }
            for module_id, rotor in self._rotor_specs:
                module_index = module_index_by_id[module_id]
                thrust_transform = world_link[rotor.thrust_frame_link]
                thrust_origin_world = thrust_transform.translation[:, module_index]
                origin_body = (
                    world_to_body
                    @ (thrust_origin_world - com_world).unsqueeze(-1)
                ).squeeze(-1)
                joint_id = rotor.vectoring_joint_ids[0]
                joint = self._joints_by_id[joint_id]
                arm_rotation_world = world_link[joint.parent_link].rotation[:, module_index]
                z_sign = (
                    1.0
                    if sum(
                        float(left) * float(right)
                        for left, right in zip(rotor.thrust_axis_local, (0.0, 0.0, 1.0))
                    )
                    >= 0.0
                    else -1.0
                )
                local_z = self._constant(
                    (0.0, 0.0, z_sign), device=device, dtype=dtype
                ).reshape(1, 3, 1)
                z_axis_world = (arm_rotation_world @ local_z).squeeze(-1)
                z_axis_body = _normalize(
                    (world_to_body @ z_axis_world.unsqueeze(-1)).squeeze(-1)
                )
                axis_module = joint_axis_module[joint_id][:, module_index]
                axis_world = (
                    module_rotation_world[:, module_index]
                    @ axis_module.unsqueeze(-1)
                ).squeeze(-1)
                vectoring_axis_body = _normalize(
                    (world_to_body @ axis_world.unsqueeze(-1)).squeeze(-1)
                )
                x_axis_body = _normalize(
                    torch.cross(vectoring_axis_body, z_axis_body, dim=-1)
                )
                x_columns.append(
                    _wrench_column(
                        origin_body,
                        x_axis_body,
                        float(rotor.reaction_torque_coeff_nm_per_n),
                    )
                )
                z_columns.append(
                    _wrench_column(
                        origin_body,
                        z_axis_body,
                        float(rotor.reaction_torque_coeff_nm_per_n),
                    )
                )
                current_angles.append(
                    local_joint_positions_rad[
                        :, module_index, self._joint_index[joint_id]
                    ]
                )
                thrust_min.append(float(rotor.thrust_min_n))
                thrust_max.append(float(rotor.thrust_max_n))
                angle_lower.append(float(joint.limit_lower or 0.0))
                angle_upper.append(float(joint.limit_upper or 0.0))
                angle_velocity.append(float(_joint_limits(joint, self.physical_model)["velocity"] or 0.0))
                rotor_module_ids.append(module_id)
                rotor_local_ids.append(rotor.rotor_id)
                vectoring_global_joint_ids.append(f"module_{module_id}:{joint_id}")

        rotor_shape = (1, self.rotor_count)
        internal_loads = (None, None, None, None, None, None)
        if self._internal_load_model is not None:
            rotor_positions = torch.stack([world_link[r.thrust_frame_link].translation[:, self.module_ids.index(m)]
                for m, r in self._rotor_specs], dim=1)
            rotor_x = (base_rotation_world[:, None] @ torch.stack(x_columns, dim=1)[..., :3, None]).squeeze(-1)
            rotor_z = (base_rotation_world[:, None] @ torch.stack(z_columns, dim=1)[..., :3, None]).squeeze(-1)
            internal_loads = self._internal_load_model.evaluate(world_link=world_link,
                module_rotation_world=module_rotation_world, joint_axis_module=joint_axis_module,
                rotor_positions_world=rotor_positions, rotor_x_axes_world=rotor_x, rotor_z_axes_world=rotor_z,
                angular_velocity_world=body_twist[:, 3:])
        return BatchedRigidBodyControlModel(
            joint_load_matrix=internal_loads[0], joint_gravity_load_nm=internal_loads[1],
            joint_mass_jacobian=internal_loads[2], joint_contact_jacobian=internal_loads[3],
            joint_angular_mass_matrix=internal_loads[4], joint_centrifugal_load_nm=internal_loads[5],
            body_pose_world=body_pose,
            body_twist_world=body_twist,
            total_mass_kg=total_mass,
            inertia_body_matrix=inertia_body_matrix,
            inertia_body=_matrix_to_inertia6(inertia_body_matrix),
            virtual_x_wrench_columns=torch.stack(x_columns, dim=1),
            virtual_z_wrench_columns=torch.stack(z_columns, dim=1),
            current_vectoring_angles_rad=torch.stack(current_angles, dim=1),
            thrust_min_n=self._constant(
                thrust_min, device=device, dtype=dtype
            ).reshape(rotor_shape).expand(batch_size, -1),
            thrust_max_n=self._constant(
                thrust_max, device=device, dtype=dtype
            ).reshape(rotor_shape).expand(batch_size, -1),
            vectoring_lower_rad=self._constant(
                angle_lower, device=device, dtype=dtype
            ).reshape(rotor_shape).expand(batch_size, -1),
            vectoring_upper_rad=self._constant(
                angle_upper, device=device, dtype=dtype
            ).reshape(rotor_shape).expand(batch_size, -1),
            vectoring_velocity_limit_radps=self._constant(
                angle_velocity, device=device, dtype=dtype
            ).reshape(rotor_shape).expand(batch_size, -1),
            rotor_module_ids=tuple(rotor_module_ids),
            rotor_local_ids=tuple(rotor_local_ids),
            vectoring_global_joint_ids=tuple(vectoring_global_joint_ids),
        )

    def _module_link_kinematics(
        self, joint_positions: torch.Tensor
    ) -> tuple[dict[str, _TensorTransform], dict[str, torch.Tensor]]:
        batch_size, module_count, _ = joint_positions.shape
        device = joint_positions.device
        dtype = joint_positions.dtype
        identity = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 3, 3)
        identity = identity.expand(batch_size, module_count, -1, -1)
        zero = torch.zeros(
            (batch_size, module_count, 3), device=device, dtype=dtype
        )
        root_transform: dict[str, _TensorTransform] = {
            root: _TensorTransform(identity, zero) for root in self._roots
        }
        joint_axis_root: dict[str, torch.Tensor] = {}
        pending = list(self._roots)
        while pending:
            parent = pending.pop(0)
            parent_transform = root_transform[parent]
            for joint in sorted(
                self._joints_by_parent.get(parent, []),
                key=lambda item: item.joint_id,
            ):
                origin_rotation = self._constant(
                    _rpy_to_matrix(joint.origin_rpy), device=device, dtype=dtype
                ).reshape(1, 1, 3, 3)
                origin_translation = self._constant(
                    joint.origin_xyz, device=device, dtype=dtype
                ).reshape(1, 1, 3)
                joint_frame = _compose(
                    parent_transform,
                    _TensorTransform(origin_rotation, origin_translation),
                )
                local_axis = self._constant(
                    joint.axis_xyz, device=device, dtype=dtype
                ).reshape(1, 1, 3, 1)
                joint_axis_root[joint.joint_id] = _normalize(
                    (joint_frame.rotation @ local_axis).squeeze(-1)
                )
                position = joint_positions[
                    :, :, self._joint_index[joint.joint_id]
                ]
                motion = _joint_motion_transform(
                    joint, position, device=device, dtype=dtype,
                    axis=local_axis.reshape(3),
                )
                root_transform[joint.child_link] = _compose(joint_frame, motion)
                pending.append(joint.child_link)
        if self._base_link not in root_transform:
            raise ValueError("batched rigid-body base link transform is missing")
        base_inverse = _inverse(root_transform[self._base_link])
        module_transform = {
            link_id: _compose(base_inverse, transform)
            for link_id, transform in root_transform.items()
        }
        joint_axis_module = {
            joint_id: _normalize(
                (base_inverse.rotation @ axis.unsqueeze(-1)).squeeze(-1)
            )
            for joint_id, axis in joint_axis_root.items()
        }
        missing = set(self.link_ids) - set(module_transform)
        if missing:
            raise ValueError(f"batched rigid-body links are unreachable: {sorted(missing)}")
        return module_transform, joint_axis_module

    def _validate_inputs(
        self,
        module_pose_world: torch.Tensor,
        module_twist_world: torch.Tensor,
        local_joint_positions_rad: torch.Tensor,
    ) -> None:
        if module_pose_world.ndim != 3 or module_pose_world.shape[1:] != (
            self.module_count,
            7,
        ):
            raise ValueError(
                "module_pose_world must have shape [batch, module_count, 7]"
            )
        expected_twist = (module_pose_world.shape[0], self.module_count, 6)
        expected_joint = (
            module_pose_world.shape[0],
            self.module_count,
            self.local_joint_count,
        )
        if tuple(module_twist_world.shape) != expected_twist:
            raise ValueError(f"module_twist_world must have shape {expected_twist}")
        if tuple(local_joint_positions_rad.shape) != expected_joint:
            raise ValueError(
                f"local_joint_positions_rad must have shape {expected_joint}"
            )
        for value in (
            module_pose_world,
            module_twist_world,
            local_joint_positions_rad,
        ):
            if (
                not value.is_floating_point()
                or value.device != module_pose_world.device
                or value.dtype != module_pose_world.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    "batched rigid-body inputs must be finite and share dtype/device"
                )


def _joint_motion_transform(
    joint: JointModel,
    position: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    axis: torch.Tensor | None = None,
) -> _TensorTransform:
    batch_size, module_count = position.shape
    identity = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 3, 3)
    identity = identity.expand(batch_size, module_count, -1, -1)
    zero = torch.zeros((batch_size, module_count, 3), device=device, dtype=dtype)
    if joint.joint_type in {"revolute", "continuous"}:
        if axis is None:
            axis = torch.tensor(joint.axis_xyz, device=device, dtype=dtype)
        rotation = _axis_angle_to_matrix(axis, position)
        return _TensorTransform(rotation, zero)
    if joint.joint_type == "prismatic":
        if axis is None:
            axis = torch.tensor(joint.axis_xyz, device=device, dtype=dtype)
        axis = axis / axis.norm().clamp_min(1.0e-12)
        return _TensorTransform(identity, position.unsqueeze(-1) * axis)
    return _TensorTransform(identity, zero)


def _compose(left: _TensorTransform, right: _TensorTransform) -> _TensorTransform:
    rotation = left.rotation @ right.rotation
    translation = left.translation + (
        left.rotation @ right.translation.unsqueeze(-1)
    ).squeeze(-1)
    return _TensorTransform(rotation, translation)


def _inverse(transform: _TensorTransform) -> _TensorTransform:
    rotation = transform.rotation.transpose(-1, -2)
    translation = -(rotation @ transform.translation.unsqueeze(-1)).squeeze(-1)
    return _TensorTransform(rotation, translation)


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1.0e-12)
    x, y, z = axis.unbind()
    zero = torch.zeros((), device=axis.device, dtype=axis.dtype)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype)
    sine = torch.sin(angle)[..., None, None]
    cosine = torch.cos(angle)[..., None, None]
    return identity + sine * skew + (1.0 - cosine) * (skew @ skew)


def _quaternion_to_matrix(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion_xyzw / quaternion_xyzw.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _wrench_column(
    origin_body: torch.Tensor, axis_body: torch.Tensor, reaction_coeff: float | torch.Tensor
) -> torch.Tensor:
    axis = _normalize(axis_body)
    torque = torch.cross(origin_body, axis, dim=-1) + reaction_coeff * axis
    return torch.cat((axis, torque), dim=-1)


def _matrix_to_inertia6(matrix: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            matrix[..., 0, 0],
            matrix[..., 0, 1],
            matrix[..., 0, 2],
            matrix[..., 1, 1],
            matrix[..., 1, 2],
            matrix[..., 2, 2],
        ),
        dim=-1,
    )


def _inertia6_to_matrix(values) -> tuple[tuple[float, float, float], ...]:
    ixx, ixy, ixz, iyy, iyz, izz = (float(value) for value in values)
    return ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz))


def _module_base_link(physical_model: PhysicalModel, link_ids: set[str]) -> str:
    metadata = physical_model.metadata.get("baselink")
    if isinstance(metadata, dict) and isinstance(metadata.get("name"), str):
        name = str(metadata["name"])
        if name in link_ids:
            return name
    if "fc" in link_ids:
        return "fc"
    roots = physical_model.metadata.get("root_links")
    if isinstance(roots, list) and roots and str(roots[0]) in link_ids:
        return str(roots[0])
    return sorted(link_ids)[0]


def _rpy_to_matrix(rpy) -> tuple[tuple[float, float, float], ...]:
    import math

    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


__all__ = [
    "BatchedRigidBodyControlModel",
    "BatchedRigidBodyControlModelBuilder",
]
