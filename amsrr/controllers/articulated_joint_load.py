from __future__ import annotations

"""Analytic generalized gravity/rotor loads for a tree of docked modules.

The model estimates internal motor demand; it does not estimate contact wrench.
Each physical joint is cut once in the complete link graph. The sign of the
moving component respects a URDF joint whose child lies toward the robot base.
"""

import torch
from amsrr.robot_model.whole_structure_kinematics import WholeStructureKinematics


def append_grasp_opening_constraints(matrix, bias, limit, normal_bias,
                                     opening_map, thrust_max):
    """Add D (C x + b - b_normal) <= 0 without restricting helpful closure.

    D maps additional motor load to outward anchor displacement using joint
    stiffness and the fixed-CoM geometric Jacobian. This is an approximate
    deflection constraint, not a contact-force estimate. Normalize rows for
    conditioning; express the one-sided bound through the allocator's existing
    symmetric intervals. The lower bound follows from physical rotor maxima.
    Zero D rows disable this constraint outside acquisition/holding.
    """
    if (opening_map.ndim != 3 or opening_map.shape[0] != matrix.shape[0]
            or opening_map.shape[-1] != matrix.shape[1]
            or normal_bias.shape != bias.shape):
        raise ValueError("invalid grasp opening constraint dimensions")
    normalized = opening_map / opening_map.norm(dim=-1, keepdim=True).clamp_min(1.e-12)
    extra_matrix = normalized @ matrix
    offset = (normalized @ (bias - normal_bias).unsqueeze(-1)).squeeze(-1)
    rotor_norm = extra_matrix.reshape(*extra_matrix.shape[:2], -1, 2).norm(dim=-1)
    bound = ((rotor_norm * thrust_max[:, None]).sum(-1) + offset.abs()).clamp_min(1.)
    return (torch.cat((matrix, extra_matrix), dim=1),
            torch.cat((bias, offset + bound / 2), dim=1),
            torch.cat((limit, bound / 2), dim=1))


class ArticulatedJointLoadModel:
    def __init__(self, morphology, physical_model, contact_references=()):
        kinematics = WholeStructureKinematics()
        model, graph = kinematics._contexts(morphology, physical_model)
        self.module_ids = tuple(sorted(graph.modules))
        self.link_ids = tuple(link.link_id for link in physical_model.links)
        self.nodes = tuple((m, l) for m in self.module_ids for l in self.link_ids)
        index = {node: i for i, node in enumerate(self.nodes)}
        adjacency = {node: [] for node in self.nodes}
        def connect(a, b):
            adjacency[a].append(b); adjacency[b].append(a)
        for m in self.module_ids:
            for j in physical_model.joints:
                connect((m, j.parent_link), (m, j.child_link))
        for edge in morphology.dock_edges:
            if edge.latch_state == 'detached':
                continue
            a = graph.ports_by_global_id[edge.src_port_id]
            b = graph.ports_by_global_id[edge.dst_port_id]
            connect((a.module_id, model.connect_link_by_port_id[a.port_local_id]),
                    (b.module_id, model.connect_link_by_port_id[b.port_local_id]))
        root = (morphology.base_module_id, model.module_base_link_id)
        parent = {root: None}; ordered = [root]
        for node in ordered:
            for child in adjacency[node]:
                if child == parent[node]:
                    continue
                if child in parent:
                    raise ValueError('internal joint load model requires a tree morphology')
                parent[child] = node; ordered.append(child)
        if len(parent) != len(self.nodes):
            raise ValueError('internal joint load model requires a connected morphology')
        descendants = {node: {node} for node in self.nodes}
        for node in reversed(ordered[1:]):
            descendants[parent[node]].update(descendants[node])
        joints = {j.joint_id: j for j in physical_model.joints}
        self.joints = tuple((m, joints[j]) for m in self.module_ids for j in model.dock_joint_ids)
        self.joint_ids = tuple(f'module_{m}:{j.joint_id}' for m, j in self.joints)
        signed = []
        for m, joint in self.joints:
            a, b = (m, joint.parent_link), (m, joint.child_link)
            moving, sign = (b, 1.) if parent[b] == a else (a, -1.)
            signed.append([sign if node in descendants[moving] else 0. for node in self.nodes])
        self.signed = signed
        self.rotors = tuple((m, r) for m in self.module_ids for r in sorted(physical_model.rotors, key=lambda r: r.rotor_id))
        self.rotor_nodes = [index[(m, r.thrust_frame_link)] for m, r in self.rotors]
        self.masses = [link.mass_kg for _ in self.module_ids for link in physical_model.links]
        self.com_local = [link.local_com for link in physical_model.links]
        self.inertias = []
        for link in physical_model.links:
            xx, xy, xz, yy, yz, zz = link.inertia_kgm2
            self.inertias.append([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        self.contacts = tuple(contact_references)
        self.contact_nodes = [index[(r.anchor.module_id, r.surface.mechanism_link_id)] for r in self.contacts]
        self.rotor_signed = [[row[i] for i in self.rotor_nodes] for row in signed]
        self.contact_signed = [[row[i] for i in self.contact_nodes] for row in signed]
        self._cache = {}

    def constant(self, value, reference):
        key = (str(reference.device), reference.dtype, repr(value))
        if key not in self._cache:
            self._cache[key] = torch.tensor(value, device=reference.device, dtype=reference.dtype)
        return self._cache[key]

    def evaluate(self, *, world_link, module_rotation_world, joint_axis_module,
                 rotor_positions_world, rotor_x_axes_world, rotor_z_axes_world,
                 angular_velocity_world=None):
        ref = module_rotation_world
        signed = self.constant(self.signed, ref)
        masses = self.constant(self.masses, ref)
        coms = torch.stack([world_link[l].translation +
            (world_link[l].rotation @ self.constant(self.com_local[i], ref).reshape(1, 1, 3, 1)).squeeze(-1)
            for i, l in enumerate(self.link_ids)], dim=2).flatten(1, 2)
        origins = []; axes = []
        for m, joint in self.joints:
            mi = self.module_ids.index(m)
            parent = world_link[joint.parent_link]
            origins.append(parent.translation[:, mi] +
                (parent.rotation[:, mi] @ self.constant(joint.origin_xyz, ref).reshape(1, 3, 1)).squeeze(-1))
            axes.append((module_rotation_world[:, mi] @ joint_axis_module[joint.joint_id][:, mi].unsqueeze(-1)).squeeze(-1))
        origin = torch.stack(origins, dim=1); axis = torch.stack(axes, dim=1)
        if ref.device.type == 'cpu':
            rotation = torch.stack([world_link[l].rotation for l in self.link_ids], dim=2)
            inertia_world = (rotation @ self.constant(self.inertias, ref)[None, None]
                             @ rotation.transpose(-1, -2)).flatten(1, 2)
            mass_jacobian, angular_mass, centrifugal = _joint_mass_moments(
                coms, inertia_world, masses, signed, origin, axis, angular_velocity_world)
        else:
            point_jacobian = torch.cross(axis[:, :, None, :], coms[:, None, :, :] - origin[:, :, None, :], dim=-1) * signed[None, :, :, None]
            mass_jacobian = (point_jacobian * masses[None, None, :, None]).sum(dim=2)
            rotation = torch.stack([world_link[l].rotation for l in self.link_ids], dim=2)
            inertia_world = (rotation @ self.constant(self.inertias, ref)[None, None]
                             @ rotation.transpose(-1, -2)).flatten(1, 2)
            com = (coms * masses[None, :, None]).sum(1) / masses.sum()
            lever = coms - com[:, None]
            angular_mass = (torch.cross(lever[:, None].expand_as(point_jacobian), point_jacobian, dim=-1)
                           * masses[None, None, :, None]).sum(2)
            angular_mass = angular_mass + ((inertia_world[:, None] @ axis[:, :, None, :, None]).squeeze(-1)
                                          * signed[None, :, :, None]).sum(2)
            centrifugal = torch.zeros_like(mass_jacobian[..., 0])
            if angular_velocity_world is not None:
                omega = angular_velocity_world[:, None].expand_as(lever)
                force = torch.cross(omega, torch.cross(omega, lever, dim=-1), dim=-1) * masses[None, :, None]
                torque = torch.cross(omega, (inertia_world @ omega[..., None]).squeeze(-1), dim=-1)
                centrifugal = (point_jacobian * force[:, None]).sum(dim=(-1, -2)) + (
                    axis[:, :, None] * torque[:, None] * signed[None, :, :, None]).sum(dim=(-1, -2))
        gravity = -9.81 * mass_jacobian[..., 2]
        arm = rotor_positions_world[:, None, :, :] - origin[:, :, None, :]
        jac = torch.cross(axis[:, :, None, :], arm, dim=-1)
        reaction = self.constant([r.reaction_torque_coeff_nm_per_n for _, r in self.rotors], ref)
        jac = jac + axis[:, :, None, :] * reaction[None, None, :, None]
        mask = self.constant(self.rotor_signed, ref)[None, :, :]
        x = (jac * rotor_x_axes_world[:, None, :, :]).sum(-1) * mask
        z = (jac * rotor_z_axes_world[:, None, :, :]).sum(-1) * mask
        contact_positions = []
        for reference in self.contacts:
            mi = self.module_ids.index(reference.anchor.module_id)
            link = world_link[reference.surface.mechanism_link_id]
            contact_positions.append(link.translation[:, mi] +
                (link.rotation[:, mi] @ self.constant(reference.anchor.local_pose[:3], ref).reshape(1, 3, 1)).squeeze(-1))
        contact_jacobian = None
        if contact_positions:
            points = torch.stack(contact_positions, dim=1)
            contact_jacobian = torch.cross(axis[:, :, None, :], points[:, None, :, :] - origin[:, :, None, :], dim=-1)
            contact_jacobian = contact_jacobian * self.constant(self.contact_signed, ref)[None, :, :, None]
        return torch.stack((x, z), dim=-1).flatten(2), gravity, mass_jacobian, contact_jacobian, angular_mass, centrifugal


def _joint_mass_moments(coms, inertia_world, masses, signed, origin, axis, omega):
    """Same joint loads via signed first/second moments, without [B,J,N,3,3].

    For r = link COM - assembly COM and d = assembly COM - joint origin,
    r x (axis x (r+d)) = axis*(r.(r+d)) - (r+d)*(r.axis).
    Aggregating those moments first avoids recomputing each link's inertia
    separately for every joint. No mass, link, or physical term is dropped.
    """
    center = (coms * masses[None, :, None]).sum(1) / masses.sum()
    lever = coms - center[:, None]
    weighted = signed * masses[None]
    total = weighted.sum(-1)
    first = weighted @ lever
    second = (weighted @ (lever[..., :, None] * lever[..., None, :]).flatten(2)).reshape(
        coms.shape[0], signed.shape[0], 3, 3)
    inertia = (signed @ inertia_world.flatten(2)).reshape(
        coms.shape[0], signed.shape[0], 3, 3)
    displacement = center[:, None] - origin
    mass_jacobian = torch.cross(axis, first + displacement * total[None, :, None], dim=-1)
    trace = second.diagonal(dim1=-2, dim2=-1).sum(-1)
    axial = (second @ axis[..., None]).squeeze(-1)
    angular_mass = (axis * (trace + (first * displacement).sum(-1))[..., None]
                    - axial - displacement * (first * axis).sum(-1)[..., None]
                    + (inertia @ axis[..., None]).squeeze(-1))
    centrifugal = torch.zeros_like(mass_jacobian[..., 0])
    if omega is not None:
        w = omega[:, None].expand_as(first)
        force = w * (w * first).sum(-1)[..., None] - w.square().sum(-1)[..., None] * first
        torque = (torch.cross((second @ w[..., None]).squeeze(-1), w, dim=-1)
                  + torch.cross(displacement, force, dim=-1)
                  + torch.cross(w, (inertia @ w[..., None]).squeeze(-1), dim=-1))
        centrifugal = (axis * torque).sum(-1)
    return mass_jacobian, angular_mass, centrifugal
