from __future__ import annotations

from dataclasses import replace

import pytest

from amsrr.geometry.pose_math import compose_pose
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.robot_model.gripper_surfaces import (
    GripperSurfaceResolutionError,
    resolve_unoccupied_gripper_surfaces,
    select_opposing_gripper_surface_pair,
    with_available_grasp_anchors,
    with_designated_grasp_anchors,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.task_spec import TaskSpec


def _representative_three_module_grasp(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = build_grasp_carry_variant_design_output(
        task,
        irg,
        physical_model,
        variant=GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    )
    return physical_model, design.target_morphology


def test_all_available_grasp_anchors_are_canonical_and_mesh_backed(grasp_carry_dict):
    from copy import deepcopy
    physical, source = _representative_three_module_grasp(grasp_carry_dict)
    before = source.to_dict()
    kwargs = dict(slot_ids=[0], max_force_n=30., max_torque_nm=4.)
    result = with_available_grasp_anchors(source, physical, **kwargs)
    reordered = deepcopy(source)
    reordered.robot_anchors = list(reversed(reordered.robot_anchors))
    for a in reordered.robot_anchors:
        if a.anchor_type == 'grasp':
            a.anchor_id += 100
    other = with_available_grasp_anchors(reordered, physical, **kwargs)
    assert [a.to_dict() for a in result.robot_anchors] == [a.to_dict() for a in other.robot_anchors]
    surfaces = {s.port_global_id:s for s in resolve_unoccupied_gripper_surfaces(source,physical)}
    anchors = [a for a in result.robot_anchors if a.anchor_type == 'grasp']
    assert len(anchors) == len(surfaces) > 2
    for anchor in anchors:
        surface = surfaces[anchor.capability['surface_port_id']]
        assert anchor.link_id == surface.mechanism_link_id
        assert anchor.local_pose == surface.grasp_contact_frame_link
    assert source.to_dict() == before


def test_upstream_designated_ports_remain_fixed_and_use_physical_geometry(grasp_carry_dict):
    physical, source = _representative_three_module_grasp(grasp_carry_dict)
    before = source.to_dict()
    surfaces = resolve_unoccupied_gripper_surfaces(source, physical)
    selected = [surfaces[-1], surfaces[0]]
    kwargs = dict(slot_ids=[0], max_force_n=30., max_torque_nm=4.)
    result = with_designated_grasp_anchors(source, physical,
        [s.port_global_id for s in selected], **kwargs)
    anchors = [a for a in result.robot_anchors if a.anchor_type == 'grasp']
    assert len(anchors) == 2
    for anchor, surface in zip(anchors, selected):
        assert anchor.capability['surface_port_id'] == surface.port_global_id
        assert anchor.module_id == surface.module_id
        assert anchor.link_id == surface.mechanism_link_id
        assert anchor.local_pose == surface.grasp_contact_frame_link
    assert source.to_dict() == before
    occupied = source.dock_edges[0].src_port_id
    for ports in ([occupied, selected[0].port_global_id], [9999, 9998],
                  [selected[0].port_global_id]*2, [True, 2]):
        with pytest.raises(ValueError):
            with_designated_grasp_anchors(source, physical, ports, **kwargs)


def test_resolves_only_unoccupied_ports_to_actual_mesh_backed_dock_links(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology = _representative_three_module_grasp(grasp_carry_dict)
    surfaces = resolve_unoccupied_gripper_surfaces(morphology, physical_model)

    occupied_edge_ports = {
        port_id
        for edge in morphology.dock_edges
        for port_id in (edge.src_port_id, edge.dst_port_id)
    }
    assert len(surfaces) == 8
    assert not occupied_edge_ports.intersection(
        surface.port_global_id for surface in surfaces
    )
    physical_ports = {port.port_id: port for port in physical_model.dock_ports}
    physical_joints = {joint.joint_id: joint for joint in physical_model.joints}
    for surface in surfaces:
        physical_port = physical_ports[surface.port_local_id]
        mechanism_joint_id = physical_port.mechanical_limits["mechanism_joint_id"]
        mechanism_joint = physical_joints[mechanism_joint_id]
        assert surface.mechanism_link_id == physical_port.parent_link
        assert surface.mechanism_joint_id == mechanism_joint_id
        assert mechanism_joint.child_link == surface.mechanism_link_id
        assert surface.connect_frame_module == physical_port.local_pose
        expected_grasp_module = compose_pose(
            physical_port.local_pose,
            physical_port.grasp_contact_frame_from_connect
            or (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        assert surface.grasp_contact_frame_module == pytest.approx(
            expected_grasp_module
        )
        if surface.port_type == "pitch_dock":
            assert surface.grasp_contact_frame_link[0] == pytest.approx(
                0.0874086
            )
        else:
            assert surface.grasp_contact_frame_module == pytest.approx(
                surface.connect_frame_module
            )
        assert surface.collision_primitives
        assert all(
            primitive.link_id == surface.mechanism_link_id
            for primitive in surface.collision_primitives
        )
        assert all(
            primitive.primitive_type == "mesh"
            and primitive.geometry_ref is not None
            and primitive.geometry_ref.lower().endswith(".stl")
            and primitive.convex_decomposition_compatible
            and primitive.requires_convex_decomposition
            for primitive in surface.collision_primitives
        )


def test_dock_edge_endpoints_are_excluded_even_if_occupied_flag_is_stale(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology = _representative_three_module_grasp(grasp_carry_dict)
    edge_port_ids = {
        port_id
        for edge in morphology.dock_edges
        for port_id in (edge.src_port_id, edge.dst_port_id)
    }
    stale_ports = [
        replace(port, occupied=False) if port.port_global_id in edge_port_ids else port
        for port in morphology.ports
    ]
    stale_morphology = replace(morphology, ports=stale_ports)

    surfaces = resolve_unoccupied_gripper_surfaces(
        stale_morphology,
        physical_model,
    )
    assert not edge_port_ids.intersection(
        surface.port_global_id for surface in surfaces
    )


def test_selects_deterministic_opposing_surfaces_on_two_grasp_arm_modules(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology = _representative_three_module_grasp(grasp_carry_dict)

    pair = select_opposing_gripper_surface_pair(morphology, physical_model)
    repeated = select_opposing_gripper_surface_pair(morphology, physical_model)

    assert pair == repeated
    assert (pair.first.module_id, pair.second.module_id) == (1, 2)
    assert pair.grasp_anchor_module_ids == (1, 2)
    assert {pair.first.port_type, pair.second.port_type} == {"yaw_dock"}
    assert {pair.first.port_global_id, pair.second.port_global_id} == {7, 10}
    assert pair.first.mechanism_joint_id != pair.second.mechanism_joint_id
    assert pair.first_inward_alignment == pytest.approx(1.0, abs=1.0e-9)
    assert pair.second_inward_alignment == pytest.approx(1.0, abs=1.0e-9)
    assert pair.opposition_alignment == pytest.approx(1.0, abs=1.0e-9)
    assert abs(pair.first_inward_axis_design[0]) < 1.0e-5
    assert (
        pair.first.mechanism_joint_limit_lower
        <= pair.first_mechanism_position_target
        <= pair.first.mechanism_joint_limit_upper
    )
    assert (
        pair.second.mechanism_joint_limit_lower
        <= pair.second_mechanism_position_target
        <= pair.second.mechanism_joint_limit_upper
    )


def test_resolver_fails_closed_when_dock_collision_is_not_mesh_compatible(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology = _representative_three_module_grasp(grasp_carry_dict)
    free_port = next(port for port in morphology.ports if not port.occupied)
    physical_port = next(
        port
        for port in physical_model.dock_ports
        if port.port_id == free_port.port_local_id
    )
    collision_primitives = [
        (
            replace(primitive, primitive_type="box", geometry_ref=None)
            if primitive.link_id == physical_port.parent_link
            else primitive
        )
        for primitive in physical_model.collision_primitives
    ]
    incompatible_model = replace(
        physical_model,
        collision_primitives=collision_primitives,
    )

    with pytest.raises(
        GripperSurfaceResolutionError,
        match="must be mesh or convex",
    ):
        resolve_unoccupied_gripper_surfaces(morphology, incompatible_model)


def test_resolver_accepts_precomputed_convex_dock_collision(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology = _representative_three_module_grasp(grasp_carry_dict)
    free_port = next(port for port in morphology.ports if not port.occupied)
    physical_port = next(
        port
        for port in physical_model.dock_ports
        if port.port_id == free_port.port_local_id
    )
    collision_primitives = [
        (
            replace(primitive, primitive_type="convex")
            if primitive.link_id == physical_port.parent_link
            else primitive
        )
        for primitive in physical_model.collision_primitives
    ]
    convex_model = replace(
        physical_model,
        collision_primitives=collision_primitives,
    )

    surface = next(
        surface
        for surface in resolve_unoccupied_gripper_surfaces(morphology, convex_model)
        if surface.port_global_id == free_port.port_global_id
    )
    assert all(
        primitive.primitive_type == "convex"
        and primitive.convex_decomposition_compatible
        and not primitive.requires_convex_decomposition
        for primitive in surface.collision_primitives
    )


def test_resolver_fails_closed_for_stale_graph_connect_frame(
    grasp_carry_dict: dict,
) -> None:
    physical_model, morphology = _representative_three_module_grasp(grasp_carry_dict)
    free_port = next(port for port in morphology.ports if not port.occupied)
    stale_pose = (
        free_port.local_pose[0] + 1.0e-4,
        *free_port.local_pose[1:],
    )
    stale_ports = [
        (
            replace(port, local_pose=stale_pose)
            if port.port_global_id == free_port.port_global_id
            else port
        )
        for port in morphology.ports
    ]
    stale_morphology = replace(morphology, ports=stale_ports)

    with pytest.raises(GripperSurfaceResolutionError, match="connect frame is stale"):
        resolve_unoccupied_gripper_surfaces(stale_morphology, physical_model)


def test_assembled_boundary_crosses_seam_and_skips_occupied_docks():
    from types import SimpleNamespace as NS
    from amsrr.robot_model.gripper_surfaces import free_dock_boundary_cycles
    physical = build_physical_model_from_config('configs/robot/robot_model.yaml')
    names = ['pitch_connect_point_1', 'pitch_connect_point_2', 'yaw_connect_point_1', 'yaw_connect_point_2']
    graph = NS(modules=[NS(module_id=0), NS(module_id=1)],
        ports=[NS(module_id=m, port_global_id=base+i, port_local_id=n, occupied=base+i in (10,23))
               for m,base in [(0,10),(1,20)] for i,n in enumerate(names)],
        dock_edges=[NS(src_port_id=10,dst_port_id=23)])
    # Authored perimeter of two joined modules, including the seam crossing.
    assert free_dock_boundary_cycles(graph, physical) == ((11,21,20,22,12,13),)
    graph.ports.reverse()
    graph.modules.reverse()
    assert free_dock_boundary_cycles(graph, physical) == ((11,21,20,22,12,13),)


def test_neighbor_anchors_have_five_members_per_group_without_duplicate_surfaces(grasp_carry_dict):
    from amsrr.robot_model.gripper_surfaces import with_neighbor_grasp_anchors
    physical, source = _representative_three_module_grasp(grasp_carry_dict)
    before = source.to_dict()
    result = with_neighbor_grasp_anchors(source, physical, slot_ids=[0], max_force_n=30., max_torque_nm=4.)
    anchors = [a for a in result.robot_anchors if a.anchor_type == 'grasp']
    assert len({a.capability['surface_port_id'] for a in anchors}) == len(anchors)
    for mask in (1,2):
        assert sum(bool(a.capability['grasp_neighborhood_mask'] & mask) for a in anchors) == 5
    for old in source.robot_anchors:
        new = next(a for a in result.robot_anchors if a.anchor_id == old.anchor_id)
        assert (new.module_id,new.link_id,new.local_pose) == (old.module_id,old.link_id,old.local_pose)
    assert source.to_dict() == before
