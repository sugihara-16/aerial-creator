"""Owned CPU arrays for the same articulated dynamics used by the tensor model.

The extension is built explicitly for the deployment host. Its identity includes
both source and compiler flags; stale or absent binaries fail before rollout.
This path supports float32/float64 without autograd. Tensor execution remains the
reference implementation and handles differentiable or GPU calls.
"""

from dataclasses import fields
from functools import lru_cache
import importlib.machinery
import importlib.util
from pathlib import Path

import numpy as np
import torch
from amsrr.utils.hashing import hash_file


@lru_cache(maxsize=1)
def load_native():
    root = Path(__file__).resolve().parents[2]
    name = "_request_dynamics_native"
    source = root / "amsrr/controllers/native/request_dynamics_native.cpp"
    script = root / "scripts/build_request_dynamics_native.sh"
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = root / "build/request_dynamics_native" / f"{name}{suffix}"
        if path.is_file():
            identity = Path(str(path) + ".build-identity")
            if not identity.is_file() or identity.read_text().splitlines() != [
                hash_file(source),
                hash_file(script),
            ]:
                raise RuntimeError(
                    f"Stale CPU dynamics extension; rebuild with {script}"
                )
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError(f"Build the CPU dynamics extension first: {script}")


class NativeCpuControlModel:
    """One fixed topology; every call consumes fresh state and owns its output."""

    def __init__(self, builder, inputs):
        from amsrr.controllers.batched_rigid_body_model import (
            _rpy_to_matrix,
            _inertia6_to_matrix,
        )

        b = builder
        dtype = inputs[0].dtype
        self.dtype = dtype
        _native_model = load_native()
        joints = list(b.physical_model.joints)
        links = list(b.physical_model.links)
        li = {l.link_id: i for i, l in enumerate(links)}
        ji = {j.joint_id: i for i, j in enumerate(joints)}
        mi = {m: i for i, m in enumerate(b.module_ids)}
        order = []
        pending = list(b._roots)
        while pending:
            parent = pending.pop(0)
            for j in sorted(
                b._joints_by_parent.get(parent, []), key=lambda j: j.joint_id
            ):
                order.append(ji[j.joint_id])
                pending.append(j.child_link)
        load = b._internal_load_model
        if load is None:
            raise ValueError("native dynamics requires internal load model")
        cfg = dict(
            modules=b.module_count,
            base_module=b.base_module_index,
            base_link=li[b._base_link],
            parents=[li[j.parent_link] for j in joints],
            children=[li[j.child_link] for j in joints],
            types=[
                (
                    1
                    if j.joint_type in ("revolute", "continuous")
                    else 2 if j.joint_type == "prismatic" else 0
                )
                for j in joints
            ],
            order=order,
            masses=[l.mass_kg for l in links],
            com_local=[l.local_com for l in links],
            inertias=[_inertia6_to_matrix(l.inertia_kgm2) for l in links],
            com_weights=[
                b._joint_com_mass_weights.get(j.joint_id, [0.0] * len(links))
                for j in joints
            ],
            axes=[j.axis_xyz for j in joints],
            origin_rotations=[_rpy_to_matrix(j.origin_rpy) for j in joints],
            origin_translations=[j.origin_xyz for j in joints],
            rotor_modules=[mi[m] for m, r in b._rotor_specs],
            rotor_links=[li[r.thrust_frame_link] for m, r in b._rotor_specs],
            rotor_joints=[ji[r.vectoring_joint_ids[0]] for m, r in b._rotor_specs],
            reactions=[r.reaction_torque_coeff_nm_per_n for m, r in b._rotor_specs],
            zsign=[
                1.0 if r.thrust_axis_local[2] >= 0 else -1.0 for m, r in b._rotor_specs
            ],
            load_modules=[mi[m] for m, j in load.joints],
            load_joints=[ji[j.joint_id] for m, j in load.joints],
            signed_load=load.signed,
            rotor_signed=load.rotor_signed,
            contact_signed=load.contact_signed,
            contact_modules=[mi[r.anchor.module_id] for r in load.contacts],
            contact_links=[li[r.surface.mechanism_link_id] for r in load.contacts],
            contact_positions=[r.anchor.local_pose[:3] for r in load.contacts],
        )
        cfg["total"] = float(
            torch.tensor(cfg["masses"], dtype=dtype).sum() * b.module_count
        )
        integer_keys = {
            "parents",
            "children",
            "types",
            "order",
            "rotor_modules",
            "rotor_links",
            "rotor_joints",
            "load_modules",
            "load_joints",
            "contact_modules",
            "contact_links",
        }
        for k, v in cfg.items():
            if isinstance(v, list) and k not in integer_keys:
                cfg[k] = np.asarray(
                    v, dtype=np.float32 if dtype == torch.float32 else np.float64
                )
        self.native = (
            _native_model.FloatModel
            if dtype == torch.float32
            else _native_model.DoubleModel
        )(cfg)
        self.template = b._build(*inputs)

    def __call__(self, *inputs):
        result = {
            k: torch.from_numpy(v)
            for k, v in self.native.evaluate(
                *(x.detach().numpy() for x in inputs)
            ).items()
        }
        from amsrr.controllers.batched_rigid_body_model import _matrix_to_inertia6

        result["inertia_body"] = _matrix_to_inertia6(result["inertia_body_matrix"])
        for f in fields(self.template):
            if f.name not in result or getattr(self.template, f.name) is None:
                result[f.name] = getattr(self.template, f.name)
        return type(self.template)(**result)
