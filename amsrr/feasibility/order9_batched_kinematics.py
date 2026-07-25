"""NumPy-batched whole-structure kinematics for the Order 9 posture resolver.

This is also the independently readable fallback/reference implementation
used to validate the native C++/Eigen kernel.  It evaluates all
finite-difference samples in one dense NumPy batch and computes the assembled
CoM in that same pass.  It intentionally omits collision queries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from amsrr.geometry.pose_math import (
    FACE_TO_FACE_DOCK_RELATION,
    quat_from_matrix,
    quat_to_matrix,
    rpy_to_matrix,
)
from amsrr.robot_model.whole_structure_kinematics import (
    EdgeConstraintResidual,
    MeshBackedAnchorReference,
    WholeStructureKinematics,
    WholeStructureKinematicsConfig,
    WholeStructureKinematicsResult,
    _build_graph_context,
    _build_model_context,
    _finite_difference_samples,
    _global_dock_joint_id,
    _ordered_global_dock_joint_ids,
    _pose_finite_difference,
    _validate_anchor_references,
    _validate_global_q,
)
from amsrr.schemas.common import Pose7D
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import JointModel, PhysicalModel


def _compose(
    left_r: np.ndarray,
    left_p: np.ndarray,
    right_r: np.ndarray,
    right_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = left_r @ right_r
    translation = left_p + np.einsum("...ij,...j->...i", left_r, right_p)
    return rotation, translation


def _inverse(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inverse_rotation = np.swapaxes(rotation, -1, -2)
    inverse_translation = -np.einsum(
        "...ij,...j->...i", inverse_rotation, translation
    )
    return inverse_rotation, inverse_translation


def _pose_arrays(pose: Pose7D) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(
        quat_to_matrix(tuple(float(value) for value in pose[3:7])),
        dtype=np.float64,
    )
    return rotation, np.asarray(pose[:3], dtype=np.float64)


def _poses_from_arrays(
    rotations: np.ndarray,
    translations: np.ndarray,
) -> list[Pose7D]:
    result: list[Pose7D] = []
    for rotation, translation in zip(rotations, translations, strict=True):
        quaternion = quat_from_matrix(
            tuple(tuple(float(value) for value in row) for row in rotation)
        )
        result.append(
            (
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
                *quaternion,
            )
        )
    return result


def _joint_motion(
    joint: JointModel,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_shape = values.shape
    identity = np.broadcast_to(np.eye(3), (*batch_shape, 3, 3)).copy()
    zero = np.zeros((*batch_shape, 3), dtype=np.float64)
    if joint.joint_type == "fixed":
        return identity, zero
    axis = np.asarray(joint.axis_xyz, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    if joint.joint_type == "prismatic":
        return identity, values[..., None] * axis
    if joint.joint_type not in {"revolute", "continuous"}:
        raise ValueError(f"unsupported joint type {joint.joint_type!r}")
    x, y, z = axis
    skew = np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=np.float64,
    )
    outer = np.outer(axis, axis)
    sine = np.sin(values)[..., None, None]
    cosine = np.cos(values)[..., None, None]
    rotation = cosine * identity + (1.0 - cosine) * outer + sine * skew
    return rotation, zero


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Stable SO(3) log for the tiny finite-difference rotations used here."""

    vector = np.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        axis=-1,
    )
    sine = 0.5 * np.linalg.norm(vector, axis=-1)
    cosine = np.clip(
        0.5 * (np.trace(rotation, axis1=-2, axis2=-1) - 1.0),
        -1.0,
        1.0,
    )
    angle = np.arctan2(sine, cosine)
    scale = np.empty_like(angle)
    small = sine <= 1.0e-12
    scale[small] = 0.5
    scale[~small] = angle[~small] / (2.0 * sine[~small])
    return vector * scale[..., None]


@dataclass(frozen=True)
class _CompiledJoint:
    joint: JointModel
    parent_link: str
    child_link: str
    origin_rotation: np.ndarray
    origin_translation: np.ndarray


@dataclass
class BatchedKinematicsStats:
    batch_evaluations: int = 0
    scalar_forward_calls: int = 0
    fixed_base_jacobian_calls: int = 0
    fixed_centroidal_jacobian_calls: int = 0
    maximum_batch_size: int = 0


class BatchedWholeStructureKinematics(WholeStructureKinematics):
    """Drop-in kinematics with batched finite differences."""

    def __init__(
        self,
        physical_model: PhysicalModel,
        config: WholeStructureKinematicsConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.physical_model = physical_model
        self.stats = BatchedKinematicsStats()
        self._compiled_morphology: MorphologyGraph | None = None
        self._model = _build_model_context(physical_model, self.config)
        self._graph = None
        self._module_ids: tuple[int, ...] = ()
        self._module_index: dict[int, int] = {}
        self._joint_ids = self._model.dock_joint_ids
        self._global_joint_ids: tuple[str, ...] = ()
        self._global_joint_column: dict[str, int] = {}
        self._compiled_joints = self._compile_full_joint_tree()
        self._links = tuple(physical_model.links)
        self._module_mass = sum(float(link.mass_kg) for link in self._links)
        if self._module_mass <= 0.0:
            raise ValueError("PhysicalModel module mass must be positive")

    def _compile_full_joint_tree(self) -> tuple[_CompiledJoint, ...]:
        outgoing: dict[str, list[JointModel]] = {}
        child_links: set[str] = set()
        for joint in self.physical_model.joints:
            outgoing.setdefault(joint.parent_link, []).append(joint)
            child_links.add(joint.child_link)
        link_ids = {link.link_id for link in self.physical_model.links}
        roots = sorted(link_ids - child_links)
        if len(roots) != 1:
            raise ValueError("PhysicalModel must have one root")
        self._physical_root_link = roots[0]
        compiled: list[_CompiledJoint] = []
        frontier = [roots[0]]
        while frontier:
            parent = frontier.pop(0)
            for joint in sorted(
                outgoing.get(parent, ()), key=lambda value: value.joint_id
            ):
                compiled.append(
                    _CompiledJoint(
                        joint=joint,
                        parent_link=parent,
                        child_link=joint.child_link,
                        origin_rotation=np.asarray(
                            rpy_to_matrix(joint.origin_rpy), dtype=np.float64
                        ),
                        origin_translation=np.asarray(
                            joint.origin_xyz, dtype=np.float64
                        ),
                    )
                )
                frontier.append(joint.child_link)
        return tuple(compiled)

    def _ensure_graph(self, morphology: MorphologyGraph) -> None:
        if morphology is self._compiled_morphology:
            return
        self._graph = _build_graph_context(
            morphology, self.physical_model, self.config
        )
        self._compiled_morphology = morphology
        self._module_ids = self._graph.modules
        self._module_index = {
            module_id: index for index, module_id in enumerate(self._module_ids)
        }
        self._global_joint_ids = _ordered_global_dock_joint_ids(
            self._module_ids, self._joint_ids
        )
        self._global_joint_column = {
            joint_id: index
            for index, joint_id in enumerate(self._global_joint_ids)
        }

    def _q_matrix(
        self,
        q_samples: Sequence[Mapping[str, float]],
    ) -> np.ndarray:
        return np.asarray(
            [
                [float(sample[joint_id]) for joint_id in self._global_joint_ids]
                for sample in q_samples
            ],
            dtype=np.float64,
        )

    def _local_q(
        self,
        q_matrix: np.ndarray,
        module_id: int,
        local_joint_id: str,
    ) -> np.ndarray:
        return q_matrix[
            :,
            self._global_joint_column[
                _global_dock_joint_id(module_id, local_joint_id)
            ],
        ]

    def _module_link_transforms(
        self,
        q_matrix: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        batch = q_matrix.shape[0]
        modules = len(self._module_ids)
        shape = (batch, modules)
        rotations: dict[str, np.ndarray] = {
            self._physical_root_link: np.broadcast_to(
                np.eye(3), (*shape, 3, 3)
            ).copy()
        }
        translations: dict[str, np.ndarray] = {
            self._physical_root_link: np.zeros((*shape, 3), dtype=np.float64)
        }
        for compiled in self._compiled_joints:
            parent_r = rotations[compiled.parent_link]
            parent_p = translations[compiled.parent_link]
            origin_r = np.broadcast_to(
                compiled.origin_rotation, (*shape, 3, 3)
            )
            origin_p = np.broadcast_to(
                compiled.origin_translation, (*shape, 3)
            )
            joint_r, joint_p = _compose(
                parent_r, parent_p, origin_r, origin_p
            )
            if compiled.joint.joint_id in self._joint_ids:
                values = np.stack(
                    [
                        self._local_q(
                            q_matrix, module_id, compiled.joint.joint_id
                        )
                        for module_id in self._module_ids
                    ],
                    axis=1,
                )
            else:
                values = np.zeros(shape, dtype=np.float64)
            motion_r, motion_p = _joint_motion(compiled.joint, values)
            rotations[compiled.child_link], translations[compiled.child_link] = (
                _compose(joint_r, joint_p, motion_r, motion_p)
            )

        base_r = rotations[self._model.module_base_link_id]
        base_p = translations[self._model.module_base_link_id]
        inverse_base_r, inverse_base_p = _inverse(base_r, base_p)
        for link_id in tuple(rotations):
            rotations[link_id], translations[link_id] = _compose(
                inverse_base_r,
                inverse_base_p,
                rotations[link_id],
                translations[link_id],
            )
        return rotations, translations

    def _evaluate_batch(
        self,
        *,
        morphology: MorphologyGraph,
        q_samples: Sequence[Mapping[str, float]],
        base_pose_world: Pose7D,
        references: Sequence[MeshBackedAnchorReference],
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        np.ndarray,
    ]:
        self._ensure_graph(morphology)
        assert self._graph is not None
        batch = len(q_samples)
        self.stats.batch_evaluations += 1
        self.stats.maximum_batch_size = max(
            self.stats.maximum_batch_size, batch
        )
        q_matrix = self._q_matrix(q_samples)
        local_r, local_p = self._module_link_transforms(q_matrix)
        base_r_scalar, base_p_scalar = _pose_arrays(base_pose_world)
        roots_r: dict[int, np.ndarray] = {
            self._graph.base_module_id: np.broadcast_to(
                base_r_scalar, (batch, 3, 3)
            ).copy()
        }
        roots_p: dict[int, np.ndarray] = {
            self._graph.base_module_id: np.broadcast_to(
                base_p_scalar, (batch, 3)
            ).copy()
        }
        frontier = [self._graph.base_module_id]
        while frontier:
            known_module = frontier.pop(0)
            known_index = self._module_index[known_module]
            for neighbor, edge in self._graph.adjacency[known_module]:
                if neighbor in roots_r:
                    continue
                child_index = self._module_index[neighbor]
                if edge.src_module_id == known_module:
                    known_port_id = edge.src_port_id
                    child_port_id = edge.dst_port_id
                    relation_pose = FACE_TO_FACE_DOCK_RELATION
                else:
                    known_port_id = edge.dst_port_id
                    child_port_id = edge.src_port_id
                    relation_r, relation_p = _pose_arrays(
                        FACE_TO_FACE_DOCK_RELATION
                    )
                    relation_r, relation_p = _inverse(
                        relation_r, relation_p
                    )
                    relation_pose = None
                known_port = self._graph.ports_by_global_id[known_port_id]
                child_port = self._graph.ports_by_global_id[child_port_id]
                known_link = self._model.connect_link_by_port_id[
                    known_port.port_local_id
                ]
                child_link = self._model.connect_link_by_port_id[
                    child_port.port_local_id
                ]
                known_connect_r, known_connect_p = _compose(
                    roots_r[known_module],
                    roots_p[known_module],
                    local_r[known_link][:, known_index],
                    local_p[known_link][:, known_index],
                )
                if relation_pose is not None:
                    relation_r_scalar, relation_p_scalar = _pose_arrays(
                        relation_pose
                    )
                    relation_r = np.broadcast_to(
                        relation_r_scalar, (batch, 3, 3)
                    )
                    relation_p = np.broadcast_to(
                        relation_p_scalar, (batch, 3)
                    )
                else:
                    relation_r = np.broadcast_to(
                        relation_r, (batch, 3, 3)
                    )
                    relation_p = np.broadcast_to(
                        relation_p, (batch, 3)
                    )
                desired_r, desired_p = _compose(
                    known_connect_r,
                    known_connect_p,
                    relation_r,
                    relation_p,
                )
                child_inverse_r, child_inverse_p = _inverse(
                    local_r[child_link][:, child_index],
                    local_p[child_link][:, child_index],
                )
                roots_r[neighbor], roots_p[neighbor] = _compose(
                    desired_r,
                    desired_p,
                    child_inverse_r,
                    child_inverse_p,
                )
                frontier.append(neighbor)

        anchors_r: dict[int, np.ndarray] = {}
        anchors_p: dict[int, np.ndarray] = {}
        for reference in references:
            module_id = reference.anchor.module_id
            module_index = self._module_index[module_id]
            link_id = reference.surface.mechanism_link_id
            link_world_r, link_world_p = _compose(
                roots_r[module_id],
                roots_p[module_id],
                local_r[link_id][:, module_index],
                local_p[link_id][:, module_index],
            )
            anchor_local_r_scalar, anchor_local_p_scalar = _pose_arrays(
                reference.anchor.local_pose
            )
            anchors_r[reference.anchor.anchor_id], anchors_p[
                reference.anchor.anchor_id
            ] = _compose(
                link_world_r,
                link_world_p,
                np.broadcast_to(anchor_local_r_scalar, (batch, 3, 3)),
                np.broadcast_to(anchor_local_p_scalar, (batch, 3)),
            )

        weighted_com = np.zeros((batch, 3), dtype=np.float64)
        for module_id in self._module_ids:
            module_index = self._module_index[module_id]
            for link in self._links:
                local_link_com = local_p[link.link_id][:, module_index] + np.einsum(
                    "bij,j->bi",
                    local_r[link.link_id][:, module_index],
                    np.asarray(link.local_com, dtype=np.float64),
                )
                world_com = roots_p[module_id] + np.einsum(
                    "bij,bj->bi", roots_r[module_id], local_link_com
                )
                weighted_com += float(link.mass_kg) * world_com
        assembled_com = weighted_com / (
            self._module_mass * len(self._module_ids)
        )
        return roots_r, roots_p, anchors_r, anchors_p, assembled_com

    def forward(
        self,
        morphology: MorphologyGraph,
        physical_model: PhysicalModel,
        global_dock_joint_positions: Mapping[str, float],
        base_pose_world: Pose7D,
        selected_anchors: Sequence[MeshBackedAnchorReference],
    ) -> WholeStructureKinematicsResult:
        if physical_model is not self.physical_model:
            raise ValueError("Batched kinematics received another model")
        self._ensure_graph(morphology)
        assert self._graph is not None
        references = _validate_anchor_references(
            morphology,
            self._graph,
            self._model,
            selected_anchors,
            self.config,
        )
        q = _validate_global_q(
            global_dock_joint_positions,
            self._global_joint_ids,
            self._graph,
            self._model,
        )
        roots_r, roots_p, anchors_r, anchors_p, _com = self._evaluate_batch(
            morphology=morphology,
            q_samples=[q],
            base_pose_world=base_pose_world,
            references=references,
        )
        self.stats.scalar_forward_calls += 1
        zeros = tuple(0.0 for _ in self._global_joint_ids)
        return WholeStructureKinematicsResult(
            module_root_poses_world={
                module_id: _poses_from_arrays(
                    roots_r[module_id], roots_p[module_id]
                )[0]
                for module_id in self._module_ids
            },
            anchor_poses_world={
                anchor_id: _poses_from_arrays(
                    anchors_r[anchor_id], anchors_p[anchor_id]
                )[0]
                for anchor_id in anchors_r
            },
            ordered_global_dock_joint_ids=self._global_joint_ids,
            anchor_jacobians={
                reference.anchor.anchor_id: (
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                )
                for reference in references
            },
            edge_constraint_residuals={},
            finite_difference_modes={},
        )

    def base_pose_for_centroidal_target(
        self,
        *,
        morphology: MorphologyGraph,
        q: Mapping[str, float],
        com_pos_world: Sequence[float],
        body_orientation_world: Sequence[float],
    ) -> Pose7D:
        """Recover the base pose directly from this kernel's batched CoM."""

        orientation_pose: Pose7D = (
            0.0,
            0.0,
            0.0,
            *tuple(float(value) for value in body_orientation_world),
        )
        _roots_r, _roots_p, _anchors_r, _anchors_p, com = (
            self._evaluate_batch(
                morphology=morphology,
                q_samples=[q],
                base_pose_world=orientation_pose,
                references=(),
            )
        )
        return (
            float(com_pos_world[0]) - float(com[0, 0]),
            float(com_pos_world[1]) - float(com[0, 1]),
            float(com_pos_world[2]) - float(com[0, 2]),
            *tuple(float(value) for value in body_orientation_world),
        )

    def compute(
        self,
        morphology: MorphologyGraph,
        physical_model: PhysicalModel,
        global_dock_joint_positions: Mapping[str, float],
        base_pose_world: Pose7D,
        selected_anchors: Sequence[MeshBackedAnchorReference],
    ) -> WholeStructureKinematicsResult:
        self._ensure_graph(morphology)
        assert self._graph is not None
        references = _validate_anchor_references(
            morphology,
            self._graph,
            self._model,
            selected_anchors,
            self.config,
        )
        q = _validate_global_q(
            global_dock_joint_positions,
            self._global_joint_ids,
            self._graph,
            self._model,
        )
        samples: list[Mapping[str, float]] = [q]
        sample_indices: list[tuple[int, int, float, str]] = []
        for joint_id in self._global_joint_ids:
            local_joint_id = joint_id.split(":", 1)[1]
            lower, upper = self._model.dock_limits[local_joint_id]
            before, after, denominator, mode = _finite_difference_samples(
                q,
                joint_id,
                lower=lower,
                upper=upper,
                requested_step=self.config.finite_difference_step,
            )
            before_index = 0
            after_index = 0
            if before is not None:
                before_index = len(samples)
                samples.append(before)
            if after is not None:
                after_index = len(samples)
                samples.append(after)
            sample_indices.append(
                (before_index, after_index, denominator, mode)
            )
        roots_r, roots_p, anchors_r, anchors_p, _com = self._evaluate_batch(
            morphology=morphology,
            q_samples=samples,
            base_pose_world=base_pose_world,
            references=references,
        )
        jacobians: dict[int, tuple[tuple[float, ...], ...]] = {}
        for reference in references:
            anchor_id = reference.anchor.anchor_id
            before_indices = np.asarray(
                [value[0] for value in sample_indices], dtype=np.intp
            )
            after_indices = np.asarray(
                [value[1] for value in sample_indices], dtype=np.intp
            )
            denominators = np.asarray(
                [value[2] for value in sample_indices], dtype=np.float64
            )
            translation = (
                anchors_p[anchor_id][after_indices]
                - anchors_p[anchor_id][before_indices]
            ) / denominators[:, None]
            delta_rotation = (
                anchors_r[anchor_id][after_indices]
                @ np.swapaxes(
                    anchors_r[anchor_id][before_indices], -1, -2
                )
            )
            angular = _rotation_log(delta_rotation) / denominators[:, None]
            matrix = np.concatenate((translation, angular), axis=1).T
            jacobians[anchor_id] = tuple(
                tuple(float(value) for value in row) for row in matrix
            )
        self.stats.fixed_base_jacobian_calls += 1
        zeros_result = self.forward(
            morphology,
            physical_model,
            q,
            base_pose_world,
            references,
        )
        return WholeStructureKinematicsResult(
            module_root_poses_world=zeros_result.module_root_poses_world,
            anchor_poses_world=zeros_result.anchor_poses_world,
            ordered_global_dock_joint_ids=self._global_joint_ids,
            anchor_jacobians=jacobians,
            edge_constraint_residuals={},
            finite_difference_modes={
                joint_id: sample_indices[index][3]
                for index, joint_id in enumerate(self._global_joint_ids)
            },
        )

    def fixed_centroidal_jacobians(
        self,
        *,
        morphology: MorphologyGraph,
        centroidal_pose_world: Pose7D,
        q: Mapping[str, float],
        limits: Mapping[str, tuple[float, float]],
        references: Sequence[MeshBackedAnchorReference],
        nominal: Mapping[int, Pose7D],
    ) -> dict[int, tuple[tuple[float, ...], ...]]:
        self._ensure_graph(morphology)
        samples: list[Mapping[str, float]] = []
        metadata: list[tuple[float, bool]] = []
        requested_step = 1.0e-5
        for joint_id in q:
            lower, upper = limits[joint_id]
            plus_room = upper - float(q[joint_id])
            minus_room = float(q[joint_id]) - lower
            perturbed = dict(q)
            if plus_room > 1.0e-12:
                step = min(requested_step, plus_room)
                perturbed[joint_id] = float(q[joint_id]) + step
                metadata.append((step, True))
            elif minus_room > 1.0e-12:
                step = min(requested_step, minus_room)
                perturbed[joint_id] = float(q[joint_id]) - step
                metadata.append((step, False))
            else:
                raise ValueError(f"joint {joint_id!r} has no difference room")
            samples.append(perturbed)
        orientation: Pose7D = (
            0.0,
            0.0,
            0.0,
            *tuple(float(value) for value in centroidal_pose_world[3:7]),
        )
        _roots_r, _roots_p, anchors_r, anchors_p, com = self._evaluate_batch(
            morphology=morphology,
            q_samples=samples,
            base_pose_world=orientation,
            references=references,
        )
        shift = (
            np.asarray(centroidal_pose_world[:3], dtype=np.float64)[None, :]
            - com
        )
        rows = {
            reference.anchor.anchor_id: [[] for _ in range(6)]
            for reference in references
        }
        for reference in references:
            anchor_id = reference.anchor.anchor_id
            nominal_r, nominal_p = _pose_arrays(nominal[anchor_id])
            perturbed_p = anchors_p[anchor_id] + shift
            denominators = np.asarray(
                [value[0] for value in metadata], dtype=np.float64
            )
            forward = np.asarray(
                [value[1] for value in metadata], dtype=bool
            )
            before_p = np.where(
                forward[:, None], nominal_p[None, :], perturbed_p
            )
            after_p = np.where(
                forward[:, None], perturbed_p, nominal_p[None, :]
            )
            before_r = np.where(
                forward[:, None, None],
                nominal_r[None, :, :],
                anchors_r[anchor_id],
            )
            after_r = np.where(
                forward[:, None, None],
                anchors_r[anchor_id],
                nominal_r[None, :, :],
            )
            translation = (
                after_p - before_p
            ) / denominators[:, None]
            angular = _rotation_log(
                after_r @ np.swapaxes(before_r, -1, -2)
            ) / denominators[:, None]
            matrix = np.concatenate((translation, angular), axis=1).T
            for axis in range(6):
                rows[anchor_id][axis].extend(
                    float(value) for value in matrix[axis]
                )
        self.stats.fixed_centroidal_jacobian_calls += 1
        return {
            anchor_id: tuple(tuple(row) for row in anchor_rows)
            for anchor_id, anchor_rows in rows.items()
        }


class BatchedCentroidalPostureIKSolverMixin:
    """Mixin overriding only the expensive fixed-centroidal Jacobian."""

    kinematics: BatchedWholeStructureKinematics

    def _fixed_centroidal_anchor_jacobians(self, **kwargs):
        return self.kinematics.fixed_centroidal_jacobians(**kwargs)


def install_batched_base_pose_patch() -> object:
    """Compatibility hook for archived batched-kernel validation scripts."""

    import amsrr.feasibility.articulated_reachability as reachability

    original = reachability.base_pose_for_centroidal_target

    def dispatched(
        morphology,
        physical_model,
        joint_positions_rad,
        com_pos_world,
        body_orientation_world,
        *,
        kinematics=None,
    ):
        if isinstance(kinematics, BatchedWholeStructureKinematics):
            if physical_model is not kinematics.physical_model:
                raise ValueError("batched base-pose model identity differs")
            return kinematics.base_pose_for_centroidal_target(
                morphology=morphology,
                q=joint_positions_rad,
                com_pos_world=com_pos_world,
                body_orientation_world=body_orientation_world,
            )
        return original(
            morphology,
            physical_model,
            joint_positions_rad,
            com_pos_world,
            body_orientation_world,
            kinematics=kinematics,
        )

    reachability.base_pose_for_centroidal_target = dispatched
    return original
