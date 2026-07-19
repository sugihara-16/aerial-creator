from __future__ import annotations

import math
from dataclasses import replace

import pytest

from amsrr.geometry.pose_math import (
    FACE_TO_FACE_DOCK_RELATION,
    compose_pose,
    inverse_pose,
)
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.robot_model.gripper_surfaces import select_opposing_gripper_surface_pair
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.robot_model.whole_structure_kinematics import (
    MeshBackedAnchorReference,
    WholeStructureKinematics,
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.task_spec import TaskSpec


def _representative_system(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = build_grasp_carry_variant_design_output(
        task,
        irg,
        physical_model,
        variant=GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    )
    morphology = design.target_morphology
    pair = select_opposing_gripper_surface_pair(morphology, physical_model)
    references = []
    for surface in (pair.first, pair.second):
        anchor = next(
            anchor
            for anchor in morphology.robot_anchors
            if anchor.capability.get("dock_port_global_id") == surface.port_global_id
        )
        references.append(MeshBackedAnchorReference(anchor=anchor, surface=surface))
    return physical_model, morphology, tuple(references)


def _zero_q(morphology, physical_model) -> dict[str, float]:
    return {
        joint_id: 0.0
        for joint_id in ordered_global_dock_joint_ids(morphology, physical_model)
    }


def _pose_error(expected, actual) -> tuple[float, float]:
    error = compose_pose(inverse_pose(expected), actual)
    translation = math.sqrt(sum(value * value for value in error[:3]))
    quaternion_norm = math.sqrt(sum(value * value for value in error[3:]))
    attitude = 2.0 * math.acos(min(1.0, abs(error[6]) / quaternion_norm))
    return translation, attitude


def test_graph_fk_satisfies_exact_connect_frames_in_either_edge_orientation(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology, references = _representative_system(grasp_carry_dict)
    base_pose = (0.2, -0.1, 0.8, 0.0, 0.0, 0.1305261922, 0.9914448614)
    solver = WholeStructureKinematics()
    result = solver.compute(
        morphology,
        physical_model,
        _zero_q(morphology, physical_model),
        base_pose,
        references,
    )

    ports = {port.port_global_id: port for port in morphology.ports}
    for edge in morphology.dock_edges:
        src = compose_pose(
            result.module_root_poses_world[edge.src_module_id],
            ports[edge.src_port_id].local_pose,
        )
        dst = compose_pose(
            result.module_root_poses_world[edge.dst_module_id],
            ports[edge.dst_port_id].local_pose,
        )
        position_error, attitude_error = _pose_error(
            compose_pose(src, FACE_TO_FACE_DOCK_RELATION),
            dst,
        )
        assert position_error < 1.0e-9
        assert attitude_error < 1.0e-8
        assert result.edge_constraint_residuals[edge.edge_id].position_error_m < 1.0e-9

    first = morphology.dock_edges[0]
    reversed_first = replace(
        first,
        src_module_id=first.dst_module_id,
        src_port_id=first.dst_port_id,
        dst_module_id=first.src_module_id,
        dst_port_id=first.src_port_id,
        relative_pose_src_to_dst=inverse_pose(first.relative_pose_src_to_dst),
    )
    reversed_graph = replace(
        morphology,
        dock_edges=[reversed_first, *morphology.dock_edges[1:]],
    )
    reversed_result = solver.compute(
        reversed_graph,
        physical_model,
        _zero_q(reversed_graph, physical_model),
        base_pose,
        references,
    )
    for anchor_id in result.anchor_poses_world:
        position_error, attitude_error = _pose_error(
            result.anchor_poses_world[anchor_id],
            reversed_result.anchor_poses_world[anchor_id],
        )
        assert position_error < 1.0e-9
        assert attitude_error < 1.0e-8


def test_commanded_dock_shape_updates_child_module_root_targets(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology, references = _representative_system(grasp_carry_dict)
    solver = WholeStructureKinematics()
    base_pose = (0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0)
    neutral_q = _zero_q(morphology, physical_model)
    neutral = solver.forward(
        morphology,
        physical_model,
        neutral_q,
        base_pose,
        references,
    )

    edge = morphology.dock_edges[0]
    graph_port = next(
        port for port in morphology.ports if port.port_global_id == edge.src_port_id
    )
    physical_port = next(
        port
        for port in physical_model.dock_ports
        if port.port_id == graph_port.port_local_id
    )
    mechanism_joint_id = str(physical_port.mechanical_limits["mechanism_joint_id"])
    commanded_q = dict(neutral_q)
    commanded_q[f"module_{edge.src_module_id}:{mechanism_joint_id}"] = 0.1
    commanded = solver.forward(
        morphology,
        physical_model,
        commanded_q,
        base_pose,
        references,
    )

    assert commanded.module_root_poses_world[
        morphology.base_module_id
    ] == pytest.approx(base_pose)
    child_position_error, child_attitude_error = _pose_error(
        neutral.module_root_poses_world[edge.dst_module_id],
        commanded.module_root_poses_world[edge.dst_module_id],
    )
    assert child_position_error > 1.0e-5 or child_attitude_error > 1.0e-5
    assert all(
        residual.position_error_m < 1.0e-9 and residual.attitude_error_rad < 1.0e-8
        for residual in commanded.edge_constraint_residuals.values()
    )


def test_jacobians_keep_all_three_module_dock_columns_and_irrelevant_zeros(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology, references = _representative_system(grasp_carry_dict)
    result = WholeStructureKinematics().compute(
        morphology,
        physical_model,
        _zero_q(morphology, physical_model),
        (0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
        references,
    )
    local_dock_ids = sorted(
        {
            str(port.mechanical_limits["mechanism_joint_id"])
            for port in physical_model.dock_ports
        }
    )
    assert len(result.ordered_global_dock_joint_ids) == 3 * len(local_dock_ids)
    assert {
        joint_id.split(":", 1)[0] for joint_id in result.ordered_global_dock_joint_ids
    } == {"module_0", "module_1", "module_2"}
    for jacobian in result.anchor_jacobians.values():
        assert len(jacobian) == 6
        assert all(
            len(row) == len(result.ordered_global_dock_joint_ids) for row in jacobian
        )

    used_port_ids = {
        port_id
        for edge in morphology.dock_edges
        for port_id in (edge.src_port_id, edge.dst_port_id)
    }
    ports = {port.port_global_id: port for port in morphology.ports}
    specs = {port.port_id: port for port in physical_model.dock_ports}
    structurally_used = {
        f"module_{ports[port_id].module_id}:"
        f"{specs[ports[port_id].port_local_id].mechanical_limits['mechanism_joint_id']}"
        for port_id in used_port_ids
    }
    selected = {
        f"module_{reference.surface.module_id}:{reference.surface.mechanism_joint_id}"
        for reference in references
    }
    irrelevant = next(
        joint_id
        for joint_id in result.ordered_global_dock_joint_ids
        if joint_id not in structurally_used | selected
    )
    column = result.ordered_global_dock_joint_ids.index(irrelevant)
    assert all(
        abs(jacobian[row][column]) < 1.0e-10
        for jacobian in result.anchor_jacobians.values()
        for row in range(6)
    )


def test_opposing_contact_wrench_mapping_has_inward_virtual_work_on_both_anchors(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology, references = _representative_system(grasp_carry_dict)
    pair = select_opposing_gripper_surface_pair(morphology, physical_model)
    result = WholeStructureKinematics().compute(
        morphology,
        physical_model,
        _zero_q(morphology, physical_model),
        (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        references,
    )
    normals = (
        pair.first_inward_axis_design,
        pair.second_inward_axis_design,
    )
    joint_count = len(result.ordered_global_dock_joint_ids)
    joint_torque = [0.0] * joint_count
    for reference, normal in zip(references, normals, strict=True):
        jacobian = result.anchor_jacobians[reference.anchor.anchor_id]
        for column in range(joint_count):
            joint_torque[column] += sum(
                jacobian[row][column] * normal[row] for row in range(3)
            )

    for reference, normal in zip(references, normals, strict=True):
        jacobian = result.anchor_jacobians[reference.anchor.anchor_id]
        induced_velocity = tuple(
            sum(
                jacobian[row][column] * joint_torque[column]
                for column in range(joint_count)
            )
            for row in range(3)
        )
        inward_virtual_work = sum(
            induced_velocity[row] * normal[row] for row in range(3)
        )
        assert inward_virtual_work > 0.0


def test_upstream_edge_mechanism_moves_downstream_selected_anchor(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology, references = _representative_system(grasp_carry_dict)
    result = WholeStructureKinematics().compute(
        morphology,
        physical_model,
        _zero_q(morphology, physical_model),
        (0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
        references,
    )
    reference = references[0]
    edge = next(
        edge
        for edge in morphology.dock_edges
        if {edge.src_module_id, edge.dst_module_id}
        == {morphology.base_module_id, reference.anchor.module_id}
    )
    base_port_id = (
        edge.src_port_id
        if edge.src_module_id == morphology.base_module_id
        else edge.dst_port_id
    )
    graph_port = next(
        port for port in morphology.ports if port.port_global_id == base_port_id
    )
    physical_port = next(
        port
        for port in physical_model.dock_ports
        if port.port_id == graph_port.port_local_id
    )
    upstream_id = (
        f"module_{morphology.base_module_id}:"
        f"{physical_port.mechanical_limits['mechanism_joint_id']}"
    )
    column = result.ordered_global_dock_joint_ids.index(upstream_id)
    jacobian = result.anchor_jacobians[reference.anchor.anchor_id]

    assert math.sqrt(sum(jacobian[row][column] ** 2 for row in range(6))) > 0.1


def test_finite_difference_is_deterministic_and_limit_safe(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology, references = _representative_system(grasp_carry_dict)
    q = _zero_q(morphology, physical_model)
    joints = {joint.joint_id: joint for joint in physical_model.joints}
    bounded_global_id = next(
        joint_id
        for joint_id in q
        if joints[joint_id.split(":", 1)[1]].limit_upper is not None
    )
    upper = joints[bounded_global_id.split(":", 1)[1]].limit_upper
    assert upper is not None
    q[bounded_global_id] = upper
    solver = WholeStructureKinematics()

    first = solver.compute(
        morphology,
        physical_model,
        q,
        (0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
        references,
    )
    repeated = solver.compute(
        morphology,
        physical_model,
        q,
        (0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
        references,
    )

    assert first == repeated
    assert first.finite_difference_modes[bounded_global_id] == "backward"
    assert all(
        math.isfinite(value)
        for jacobian in first.anchor_jacobians.values()
        for row in jacobian
        for value in row
    )
    invalid_q = dict(q)
    invalid_q[bounded_global_id] = upper + 1.0e-3
    with pytest.raises(SchemaValidationError, match="above its upper limit"):
        solver.compute(
            morphology,
            physical_model,
            invalid_q,
            (0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
            references,
        )


def test_fails_closed_for_disconnected_graph_and_unknown_or_vectoring_q(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology, references = _representative_system(grasp_carry_dict)
    solver = WholeStructureKinematics()
    base_pose = (0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0)
    q = _zero_q(morphology, physical_model)
    disconnected = replace(morphology, dock_edges=morphology.dock_edges[:-1])
    with pytest.raises(SchemaValidationError, match="tree"):
        solver.compute(disconnected, physical_model, q, base_pose, references)

    missing = dict(q)
    missing.pop(next(iter(missing)))
    with pytest.raises(SchemaValidationError, match="q map mismatch"):
        solver.compute(morphology, physical_model, missing, base_pose, references)

    vectoring = dict(q)
    vectoring["module_0:gimbal1"] = 0.0
    with pytest.raises(SchemaValidationError, match="Vectoring joints"):
        solver.compute(morphology, physical_model, vectoring, base_pose, references)
