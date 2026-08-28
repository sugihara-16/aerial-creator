"""Python adapter for the production Order 9 C++/Eigen posture IK kernel."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from amsrr.geometry.pose_math import FACE_TO_FACE_DOCK_RELATION
from amsrr.feasibility.articulated_reachability import (
    CentroidalPostureIKSolution,
    CentroidalPostureIKSolver,
    _clip_joint_map,
    _global_joint_limits,
    _joint_limits_with_normalized_reserve,
    _global_pitch_joint_ids,
    _validate_pose7d,
    resolve_mesh_backed_anchor_references,
)
from amsrr.robot_model.whole_structure_kinematics import (
    MeshBackedAnchorReference,
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel

from amsrr.feasibility.order9_batched_kinematics import (
    BatchedWholeStructureKinematics,
    _inverse,
    _pose_arrays,
    _poses_from_arrays,
)
from amsrr.feasibility.order9_native_loader import load_order9_posture_native


class CppWholeStructureKinematics(BatchedWholeStructureKinematics):
    """Use C++ for batched FK/CoM and Python for solver integration."""

    def __init__(
        self,
        physical_model: PhysicalModel,
        *,
        enable_collision_geometry: bool = False,
        enable_exact_collision_geometry: bool | None = None,
        nominal_collision_pair_manifest: Path | None = None,
    ) -> None:
        super().__init__(physical_model)
        self._cpp_kernel = None
        self._cpp_morphology = None
        self._enable_collision_geometry = bool(enable_collision_geometry)
        self._enable_exact_collision_geometry = bool(
            self._enable_collision_geometry
            if enable_exact_collision_geometry is None
            else enable_exact_collision_geometry
        )
        self._nominal_collision_pair_manifest = nominal_collision_pair_manifest
        if (
            self._enable_exact_collision_geometry
            and not self._enable_collision_geometry
        ):
            raise ValueError("exact collision geometry requires collision geometry")
        self.collision_geometry = None
        self._link_index = {
            link.link_id: index for index, link in enumerate(self.physical_model.links)
        }

    def _ensure_graph(self, morphology: MorphologyGraph) -> None:
        super()._ensure_graph(morphology)
        if morphology is self._cpp_morphology:
            return
        assert self._graph is not None
        module_index = self._module_index
        edge_parent_modules: list[int] = []
        edge_child_modules: list[int] = []
        edge_parent_links: list[int] = []
        edge_child_links: list[int] = []
        edge_relation_r: list[np.ndarray] = []
        edge_relation_p: list[np.ndarray] = []
        visited = {self._graph.base_module_id}
        frontier = [self._graph.base_module_id]
        while frontier:
            parent = frontier.pop(0)
            for child, edge in self._graph.adjacency[parent]:
                if child in visited:
                    continue
                visited.add(child)
                frontier.append(child)
                if edge.src_module_id == parent:
                    parent_port_id = edge.src_port_id
                    child_port_id = edge.dst_port_id
                    relation_r, relation_p = _pose_arrays(FACE_TO_FACE_DOCK_RELATION)
                else:
                    parent_port_id = edge.dst_port_id
                    child_port_id = edge.src_port_id
                    relation_r, relation_p = _inverse(
                        *_pose_arrays(FACE_TO_FACE_DOCK_RELATION)
                    )
                parent_port = self._graph.ports_by_global_id[parent_port_id]
                child_port = self._graph.ports_by_global_id[child_port_id]
                parent_link = self._model.connect_link_by_port_id[
                    parent_port.port_local_id
                ]
                child_link = self._model.connect_link_by_port_id[
                    child_port.port_local_id
                ]
                edge_parent_modules.append(module_index[parent])
                edge_child_modules.append(module_index[child])
                edge_parent_links.append(self._link_index[parent_link])
                edge_child_links.append(self._link_index[child_link])
                edge_relation_r.append(relation_r)
                edge_relation_p.append(relation_p)

        joint_type = {"fixed": 0, "revolute": 1, "continuous": 1, "prismatic": 2}
        native_module = load_order9_posture_native()
        self._cpp_kernel = native_module.Kernel(
            len(self._module_ids),
            len(self.physical_model.links),
            len(self._joint_ids),
            self._link_index[self._physical_root_link],
            self._link_index[self._model.module_base_link_id],
            module_index[self._graph.base_module_id],
            np.asarray(
                [
                    self._link_index[value.parent_link]
                    for value in self._compiled_joints
                ],
                dtype=np.int32,
            ),
            np.asarray(
                [self._link_index[value.child_link] for value in self._compiled_joints],
                dtype=np.int32,
            ),
            np.asarray(
                [joint_type[value.joint.joint_type] for value in self._compiled_joints],
                dtype=np.int32,
            ),
            np.asarray(
                [
                    (
                        self._joint_ids.index(value.joint.joint_id)
                        if value.joint.joint_id in self._joint_ids
                        else -1
                    )
                    for value in self._compiled_joints
                ],
                dtype=np.int32,
            ),
            np.asarray(
                [value.origin_rotation for value in self._compiled_joints],
                dtype=np.float64,
            ),
            np.asarray(
                [value.origin_translation for value in self._compiled_joints],
                dtype=np.float64,
            ),
            np.asarray(
                [value.joint.axis_xyz for value in self._compiled_joints],
                dtype=np.float64,
            ),
            np.asarray(edge_parent_modules, dtype=np.int32),
            np.asarray(edge_child_modules, dtype=np.int32),
            np.asarray(edge_parent_links, dtype=np.int32),
            np.asarray(edge_child_links, dtype=np.int32),
            np.asarray(edge_relation_r, dtype=np.float64).reshape((-1, 3, 3)),
            np.asarray(edge_relation_p, dtype=np.float64).reshape((-1, 3)),
            np.asarray(
                [link.mass_kg for link in self.physical_model.links],
                dtype=np.float64,
            ),
            np.asarray(
                [link.local_com for link in self.physical_model.links],
                dtype=np.float64,
            ),
        )
        if self._enable_collision_geometry:
            from amsrr.feasibility.order9_posture_collision import (
                configure_kernel_collision_geometry,
            )

            self.collision_geometry = configure_kernel_collision_geometry(
                kernel=self._cpp_kernel,
                physical_model=self.physical_model,
                link_index=self._link_index,
                repository_root=Path(__file__).resolve().parents[2],
                load_exact_geometry=(self._enable_exact_collision_geometry),
                nominal_collision_pair_manifest=(self._nominal_collision_pair_manifest),
            )
        self._cpp_morphology = morphology

    def _evaluate_batch(
        self,
        *,
        morphology: MorphologyGraph,
        q_samples: Sequence[Mapping[str, float]],
        base_pose_world: Pose7D,
        references: Sequence[MeshBackedAnchorReference],
    ):
        self._ensure_graph(morphology)
        assert self._cpp_kernel is not None
        batch = len(q_samples)
        self.stats.batch_evaluations += 1
        self.stats.maximum_batch_size = max(self.stats.maximum_batch_size, batch)
        q_matrix = self._q_matrix(q_samples).reshape(
            batch, len(self._module_ids), len(self._joint_ids)
        )
        base_r, base_p = _pose_arrays(base_pose_world)
        anchor_modules = np.asarray(
            [
                self._module_index[reference.anchor.module_id]
                for reference in references
            ],
            dtype=np.int32,
        )
        anchor_links = np.asarray(
            [
                self._link_index[reference.surface.mechanism_link_id]
                for reference in references
            ],
            dtype=np.int32,
        )
        local = [_pose_arrays(reference.anchor.local_pose) for reference in references]
        local_r = np.asarray([value[0] for value in local], dtype=np.float64).reshape(
            (-1, 3, 3)
        )
        local_p = np.asarray([value[1] for value in local], dtype=np.float64).reshape(
            (-1, 3)
        )
        roots_r, roots_p, anchors_r, anchors_p, com = self._cpp_kernel.evaluate(
            q_matrix,
            base_r,
            base_p,
            anchor_modules,
            anchor_links,
            local_r,
            local_p,
        )
        return (
            {
                module_id: roots_r[:, index]
                for index, module_id in enumerate(self._module_ids)
            },
            {
                module_id: roots_p[:, index]
                for index, module_id in enumerate(self._module_ids)
            },
            {
                reference.anchor.anchor_id: anchors_r[:, index]
                for index, reference in enumerate(references)
            },
            {
                reference.anchor.anchor_id: anchors_p[:, index]
                for index, reference in enumerate(references)
            },
            com,
        )


def install_cpp_centroidal_pose_patch(
    kinematics: CppWholeStructureKinematics,
) -> object:
    """Use the native kernel for the solver's remaining scalar CoM queries."""

    import amsrr.feasibility.articulated_reachability as reachability

    original = reachability._centroidal_pose

    def centroidal_pose(
        morphology,
        physical_model,
        q,
        module_root_poses,
        _builder,
    ):
        if physical_model is not kinematics.physical_model:
            return original(
                morphology,
                physical_model,
                q,
                module_root_poses,
                _builder,
            )
        base_pose = module_root_poses[morphology.base_module_id]
        _roots_r, _roots_p, _anchors_r, _anchors_p, com = kinematics._evaluate_batch(
            morphology=morphology,
            q_samples=[q],
            base_pose_world=base_pose,
            references=(),
        )
        return (
            float(com[0, 0]),
            float(com[0, 1]),
            float(com[0, 2]),
            float(base_pose[3]),
            float(base_pose[4]),
            float(base_pose[5]),
            float(base_pose[6]),
        )

    reachability._centroidal_pose = centroidal_pose
    return original


class NativeCentroidalPostureIKSolver(CentroidalPostureIKSolver):
    """Run the complete deterministic IK iteration inside C++/Eigen."""

    solver_version = "centroidal_posture_ik_cpp_eigen_v3_continuous_seed"

    def __init__(
        self,
        physical_model: PhysicalModel,
        *,
        kinematics: CppWholeStructureKinematics,
        config=None,
        collision_config=None,
    ) -> None:
        super().__init__(
            physical_model,
            config=config,
            kinematics=kinematics,
        )
        self.kinematics = kinematics
        self.native_solve_calls = 0
        self.collision_config = collision_config
        self.last_native_collision_metrics = None

    def solve(
        self,
        *,
        morphology,
        centroidal_pose_world,
        anchor_pose_targets_world,
        initial_joint_positions_rad=None,
    ):
        _validate_pose7d(centroidal_pose_world, "centroidal_pose_world")
        self.kinematics._ensure_graph(morphology)
        assert self.kinematics._cpp_kernel is not None
        ordered_ids = ordered_global_dock_joint_ids(morphology, self.physical_model)
        limits = _global_joint_limits(morphology, self.physical_model, ordered_ids)
        seed_limits = _joint_limits_with_normalized_reserve(
            limits,
            self.config.minimum_normalized_joint_limit_reserve,
        )
        reference_q = {
            joint_id: float(
                0.0
                if initial_joint_positions_rad is None
                else initial_joint_positions_rad.get(joint_id, 0.0)
            )
            for joint_id in ordered_ids
        }
        reference_q = _clip_joint_map(reference_q, seed_limits)
        target_ids = tuple(sorted(int(value) for value in anchor_pose_targets_world))
        targets = {
            anchor_id: tuple(
                float(value) for value in anchor_pose_targets_world[anchor_id]
            )
            for anchor_id in target_ids
        }
        for anchor_id, target in targets.items():
            _validate_pose7d(target, f"anchor_pose_targets_world[{anchor_id}]")
        references = resolve_mesh_backed_anchor_references(
            morphology, self.physical_model, target_ids
        )
        cache_key = self._cache_key(
            morphology=morphology,
            centroidal_pose_world=centroidal_pose_world,
            targets=targets,
            initial_q=reference_q,
        )
        cached = self._solution_cache.get(cache_key)
        solve_start_q = reference_q if cached is None else cached[0]
        cached_iterations = None if cached is None else int(cached[1])

        module_count = len(self.kinematics._module_ids)
        local_count = len(self.kinematics._joint_ids)
        q_array = np.asarray(
            [solve_start_q[joint_id] for joint_id in ordered_ids],
            dtype=np.float64,
        ).reshape(module_count, local_count)
        lower = np.asarray(
            [limits[joint_id][0] for joint_id in ordered_ids],
            dtype=np.float64,
        ).reshape(module_count, local_count)
        upper = np.asarray(
            [limits[joint_id][1] for joint_id in ordered_ids],
            dtype=np.float64,
        ).reshape(module_count, local_count)
        pitch_ids = _global_pitch_joint_ids(ordered_ids, self.physical_model)
        pitch_mask = np.asarray(
            [1 if joint_id in pitch_ids else 0 for joint_id in ordered_ids],
            dtype=np.int32,
        ).reshape(module_count, local_count)
        centroidal_r, centroidal_p = _pose_arrays(tuple(centroidal_pose_world))
        anchor_modules = np.asarray(
            [
                self.kinematics._module_index[reference.anchor.module_id]
                for reference in references
            ],
            dtype=np.int32,
        )
        anchor_links = np.asarray(
            [
                self.kinematics._link_index[reference.surface.mechanism_link_id]
                for reference in references
            ],
            dtype=np.int32,
        )
        local = [_pose_arrays(reference.anchor.local_pose) for reference in references]
        local_r = np.asarray([value[0] for value in local], dtype=np.float64).reshape(
            (-1, 3, 3)
        )
        local_p = np.asarray([value[1] for value in local], dtype=np.float64).reshape(
            (-1, 3)
        )
        target_arrays = [
            _pose_arrays(targets[reference.anchor.anchor_id])
            for reference in references
        ]
        target_r = np.asarray(
            [value[0] for value in target_arrays], dtype=np.float64
        ).reshape((-1, 3, 3))
        target_p = np.asarray(
            [value[1] for value in target_arrays], dtype=np.float64
        ).reshape((-1, 3))
        config = {
            name: getattr(self.config, name)
            for name in (
                "maximum_iterations",
                "relaxed_seed_maximum_iterations",
                "feasible_refinement_iterations",
                "damping",
                "position_weight",
                "attitude_weight",
                "continuity_regularization_weight",
                "pitch_joint_regularization_weight",
                "finite_difference_step_rad",
                "maximum_joint_step_rad",
                "maximum_base_translation_step_m",
                "maximum_base_rotation_step_rad",
                "anchor_position_tolerance_m",
                "anchor_attitude_tolerance_rad",
                "use_relaxed_seed",
            )
        }
        if self.collision_config is not None:
            config.update(self.collision_config.to_native_dict())
        native = self.kinematics._cpp_kernel.solve_centroidal(
            q_array,
            lower,
            upper,
            pitch_mask,
            centroidal_r,
            centroidal_p,
            anchor_modules,
            anchor_links,
            local_r,
            local_p,
            target_r,
            target_p,
            config,
        )
        self.native_solve_calls += 1
        self.last_native_collision_metrics = {
            "minimum_proxy_clearance_m": float(native["minimum_proxy_clearance"]),
            "active_proxy_pair_count": int(native["active_proxy_pair_count"]),
        }
        solved_q_values = np.asarray(native["q"]).reshape(-1)
        solved_q = {
            joint_id: float(solved_q_values[index])
            for index, joint_id in enumerate(ordered_ids)
        }
        base_pose = _poses_from_arrays(
            np.asarray(native["base_r"])[None, :, :],
            np.asarray(native["base_p"])[None, :],
        )[0]
        native_anchor_r = np.asarray(native["anchor_r"])
        native_anchor_p = np.asarray(native["anchor_p"])
        anchor_poses = {
            reference.anchor.anchor_id: _poses_from_arrays(
                native_anchor_r[index][None, :, :],
                native_anchor_p[index][None, :],
            )[0]
            for index, reference in enumerate(references)
        }
        com = np.asarray(native["com"])
        centroidal = (
            float(com[0]),
            float(com[1]),
            float(com[2]),
            *base_pose[3:7],
        )
        solution = CentroidalPostureIKSolution(
            feasible=bool(native["feasible"]),
            joint_positions_rad=solved_q,
            base_pose_world=base_pose,
            centroidal_pose_world=centroidal,
            anchor_poses_world=anchor_poses,
            maximum_position_error_m=float(native["maximum_position_error"]),
            maximum_attitude_error_rad=float(native["maximum_attitude_error"]),
            iterations=(
                int(native["iterations"])
                if cached_iterations is None
                else cached_iterations
            ),
            solver_version=self.solver_version,
            cache_hit=cached_iterations is not None,
        )
        if cached_iterations is not None and not solution.feasible:
            self._solution_cache.pop(cache_key, None)
            return self.solve(
                morphology=morphology,
                centroidal_pose_world=centroidal_pose_world,
                anchor_pose_targets_world=anchor_pose_targets_world,
                initial_joint_positions_rad=initial_joint_positions_rad,
            )
        if solution.feasible:
            self._remember(
                cache_key,
                solution.joint_positions_rad,
                iterations=solution.iterations,
            )
        elif self.config.pitch_joint_regularization_weight > 0.0:
            bootstrap = self.__class__(
                self.physical_model,
                kinematics=self.kinematics,
                config=replace(
                    self.config,
                    pitch_joint_regularization_weight=0.0,
                ),
                collision_config=self.collision_config,
            ).solve(
                morphology=morphology,
                centroidal_pose_world=centroidal_pose_world,
                anchor_pose_targets_world=anchor_pose_targets_world,
                initial_joint_positions_rad=initial_joint_positions_rad,
            )
            if bootstrap.feasible:
                return self.solve(
                    morphology=morphology,
                    centroidal_pose_world=centroidal_pose_world,
                    anchor_pose_targets_world=anchor_pose_targets_world,
                    initial_joint_positions_rad=(bootstrap.joint_positions_rad),
                )
        return solution
