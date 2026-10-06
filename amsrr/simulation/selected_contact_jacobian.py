"""Shift only requested PhysX COM Jacobians to link origins.

Uses the same Warp operations as Isaac Lab's shift_jacobian_com_to_origin.
Selecting the bodies before that calculation avoids shifting hundreds of
unused links. Returned tensors are newly owned on every call.
"""

import torch
import warp as wp


@wp.kernel
def shift_selected(
    pose: wp.array2d(dtype=wp.transformf),
    com: wp.array2d(dtype=wp.vec3f),
    indices: wp.array(dtype=wp.int64),
    offset: wp.int32,
    src: wp.array4d(dtype=wp.float32),
    dst: wp.array4d(dtype=wp.float32),
):
    n, c, dof = wp.tid()
    full = wp.int32(indices[c])
    b = full - offset
    r = wp.quat_rotate(wp.transform_get_rotation(pose[n, full]), com[n, full])
    v = wp.vec3(src[n, b, 0, dof], src[n, b, 1, dof], src[n, b, 2, dof])
    w = wp.vec3(src[n, b, 3, dof], src[n, b, 4, dof], src[n, b, 5, dof])
    shifted = v - wp.cross(w, r)
    for i in range(3):
        dst[n, c, i, dof] = shifted[i]
        dst[n, c, i + 3, dof] = w[i]


def selected_link_jacobians(data, indices):
    src = data.body_com_jacobian_w
    pose = data.body_link_pose_w
    offset = pose.shape[1] - src.shape[1]
    if offset not in (0, 1) or indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("invalid selected articulation Jacobian layout")
    if indices.numel() == 0 or bool(
        ((indices < offset) | (indices >= pose.shape[1])).any()
    ):
        raise ValueError("selected body has no dynamic Jacobian row")
    out = torch.empty(
        (src.shape[0], indices.numel(), 6, src.shape[-1]), device=indices.device
    )
    wp.launch(
        shift_selected,
        dim=(src.shape[0], indices.numel(), src.shape[-1]),
        inputs=[
            pose.warp,
            data.body_com_pos_b.warp,
            wp.from_torch(indices),
            offset,
            src.warp,
        ],
        outputs=[wp.from_torch(out)],
        device=str(indices.device),
    )
    return out


@wp.kernel(enable_backward=False)
def _selected_point_kernel(
    pose: wp.array2d(dtype=wp.transformf),
    points: wp.array3d(dtype=wp.float32),
    parents: wp.array(dtype=wp.int32),
    origin: wp.array(dtype=wp.vec3f),
    axes: wp.array(dtype=wp.vec3f),
    signs: wp.array2d(dtype=wp.float32),
    result: wp.array4d(dtype=wp.float32),
):
    n, c, j = wp.tid()
    t = pose[n, parents[j]]
    o = wp.transform_point(t, origin[j])
    a = wp.quat_rotate(wp.transform_get_rotation(t), axes[j])
    p = wp.vec3(points[n, c, 0], points[n, c, 1], points[n, c, 2])
    v = wp.cross(a, p - o) * signs[c, j]
    for k in range(3):
        result[n, c, k, j] = v[k]


class SelectedPointJacobian:
    def __init__(self, data, bodies, columns, *, stage=None):
        _validate_selected_indices(bodies, columns)
        from pxr import Usd, UsdPhysics, Gf, Sdf

        if stage is None:
            import isaaclab.sim as sim

            stage = sim.get_current_stage()
        view = data._root_view
        paths = view.link_paths[0]
        idx = {str(p): i for i, p in enumerate(paths)}
        dofs = view.dof_paths[0]
        root = stage.GetPrimAtPath(Sdf.Path(dofs[0]).GetParentPath())
        adjacency = {i: [] for i in range(len(paths))}
        joint_info = {}
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            q = UsdPhysics.Joint(prim)
            aa = q.GetBody0Rel().GetTargets()
            bb = q.GetBody1Rel().GetTargets()
            if len(aa) != 1 or len(bb) != 1:
                continue

            def resolve(path):
                original = str(path)
                while str(path) not in idx and not path.IsAbsoluteRootPath():
                    path = path.GetParentPath()
                if not prim.IsA(UsdPhysics.FixedJoint) and original not in idx:
                    raise ValueError("active joint must name physical bodies")
                return idx[str(path)]

            a, b = resolve(aa[0]), resolve(bb[0])
            if a == b:
                continue
            adjacency[a].append(b)
            adjacency[b].append(a)
            joint_info[str(prim.GetPath())] = (a, b, q)
        parent = {0: None}
        order = [0]
        for a in order:
            for b in adjacency[a]:
                if b == parent[a]:
                    continue
                if b in parent:
                    raise ValueError("joint tree cycle")
                parent[b] = a
                order.append(b)
        if len(parent) != len(paths):
            raise ValueError("selected Jacobian requires a connected articulation tree")
        descendants = {a: {a} for a in order}
        for a in reversed(order[1:]):
            descendants[parent[a]].update(descendants[a])
        offset = data.body_com_jacobian_w.shape[-1] - len(dofs)
        if (
            bodies.ndim != 1
            or columns.ndim != 1
            or bodies.numel() == 0
            or columns.numel() == 0
        ):
            raise ValueError("selected Jacobian needs one-dimensional nonempty indices")
        if bool(((bodies < 0) | (bodies >= len(paths))).any()) or bool(
            ((columns < offset) | (columns >= len(dofs) + offset)).any()
        ):
            raise ValueError(
                "selected Jacobian index out of bounds or floating-root column"
            )
        parents = []
        origins = []
        axes = []
        moving = []
        for column in columns.tolist():
            a, b, q = joint_info[dofs[column - offset]]
            prim = q.GetPrim()
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                raise ValueError("sparse point Jacobian supports revolute columns only")
            axis_name = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
            basis = {
                "X": Gf.Vec3f(1, 0, 0),
                "Y": Gf.Vec3f(0, 1, 0),
                "Z": Gf.Vec3f(0, 0, 1),
            }[axis_name]
            rot = q.GetLocalRot0Attr().Get()
            axis = rot.Transform(basis)
            origins.append(list(q.GetLocalPos0Attr().Get()))
            axes.append(list(axis))
            parents.append(a)
            moving.append(
                (descendants[b], 1.0) if parent[b] == a else (descendants[a], -1.0)
            )
        sign = [
            [s if i in nodes else 0.0 for nodes, s in moving] for i in bodies.tolist()
        ]
        self.parents = wp.array(parents, dtype=wp.int32, device="cpu")
        self.origins = wp.array(origins, dtype=wp.vec3f, device="cpu")
        self.axes = wp.array(axes, dtype=wp.vec3f, device="cpu")
        self.signs = wp.array(sign, dtype=wp.float32, device="cpu")
        self.width = len(parents)

    def __call__(self, data, points):
        if (
            points.ndim != 3
            or points.shape[1:] != (self.signs.shape[0], 3)
            or points.device.type != "cpu"
            or points.dtype != torch.float32
            or points.shape[0] != data.body_link_pose_w.shape[0]
        ):
            raise ValueError("invalid selected point Jacobian sample")
        n, c, _ = points.shape
        result = torch.empty(
            (n, c, 3, self.width), dtype=points.dtype, device=points.device
        )
        wp.launch(
            _selected_point_kernel,
            dim=(n, c, self.width),
            inputs=[
                data.body_link_pose_w.warp,
                wp.from_torch(points),
                self.parents,
                self.origins,
                self.axes,
                self.signs,
            ],
            outputs=[wp.from_torch(result)],
            device="cpu",
        )
        return result


def _validate_selected_indices(bodies, columns):
    if any(
        x.ndim != 1 or x.numel() == 0 or x.dtype != torch.int64
        for x in (bodies, columns)
    ):
        raise ValueError("selected Jacobian indices must be nonempty int64 vectors")


def selected_point_jacobians(data, bodies, joint_columns, point_position_world):
    """Exact tree joint Jacobians from measured link poses and authored axes.

    PhysX's dense Jacobian is retained for non-CPU calls. The sparse CPU path
    has been compared directly to it, including re-rooted module joints.
    Floating-base columns are intentionally outside this joint-only interface.
    """
    _validate_selected_indices(bodies, joint_columns)
    if point_position_world.device.type != "cpu":
        from amsrr.training.order9_anchor_normal_force_estimator import (
            shift_link_origin_linear_jacobian_to_point,
        )

        selected = selected_link_jacobians(data, bodies)
        return shift_link_origin_linear_jacobian_to_point(
            body_link_jacobian_world=selected,
            body_position_world=data.body_link_pose_w.torch.index_select(1, bodies)[
                ..., :3
            ],
            point_position_world=point_position_world,
            joint_columns=joint_columns,
        )
    key = (tuple(bodies.tolist()), tuple(joint_columns.tolist()), id(data._root_view))
    cached = getattr(data, "_request_selected_point_jacobian", None)
    if cached is None or cached[0] != key:
        cached = (key, SelectedPointJacobian(data, bodies, joint_columns))
        data._request_selected_point_jacobian = cached
    return cached[1](data, point_position_world)
