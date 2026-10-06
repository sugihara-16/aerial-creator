"""Numerical equivalence and stale-input protection for planner acceleration."""
import os
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from amsrr.feasibility.order9_posture_collision import _mesh_geometry
from amsrr.feasibility.order9_native_posture_ik import CppWholeStructureKinematics
from amsrr.feasibility.articulated_reachability import (
    _centroidal_pose, base_pose_for_centroidal_target,
    resolve_mesh_backed_anchor_references,
)
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics, ordered_global_dock_joint_ids,
)
from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from tests.unit.training.test_order9_articulated_teacher import _system


def _tetrahedron(path, size):
    points = np.array([[0,0,0],[size,0,0],[0,size,0],[0,0,size]], dtype=float)
    data = bytearray(80) + struct.pack('<I', 4)
    for face in ((0,2,1),(0,1,3),(0,3,2),(1,2,3)):
        data += struct.pack('<12fH', 0.,0.,0., *points[list(face)].ravel(), 0)
    path.write_bytes(data)


def test_mesh_cache_tracks_content_and_scale_not_path_or_timestamp(tmp_path):
    path=tmp_path/'shape.stl';_tetrahedron(path,1.)
    digest, first=_mesh_geometry(path,np.ones(3));stamp=path.stat()
    again_digest, again=_mesh_geometry(path,np.ones(3))
    assert digest==again_digest and again is first
    with pytest.raises(ValueError): first[2][0,0]=99.
    _tetrahedron(path,2.);os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    changed_digest, changed=_mesh_geometry(path,np.ones(3))
    assert changed_digest!=digest
    np.testing.assert_array_equal(changed[1],[2.,2.,2.])
    _,scaled=_mesh_geometry(path,np.array([.5,1.,2.]))
    np.testing.assert_array_equal(scaled[1],[1.,2.,4.])
    path.write_bytes(b'invalid')
    with pytest.raises(ValueError,match='truncated'): _mesh_geometry(path,np.ones(3))


def test_native_planning_geometry_matches_reference_and_recomputes_state(grasp_carry_dict):
    _, physical, context=_system(grasp_carry_dict)
    morphology=context.morphology_graph
    refs=resolve_mesh_backed_anchor_references(morphology,physical,
        [a.anchor_id for a in morphology.robot_anchors])
    ids=ordered_global_dock_joint_ids(morphology,physical)
    old=WholeStructureKinematics();new=CppWholeStructureKinematics(physical)
    builder=RigidBodyControlModelBuilder()
    base=(.4,-.2,.8,0.,0.,np.sin(.15),np.cos(.15))
    values=[]
    for angle in (0.,.2,-.15,0.):
        q={j:angle for j in ids}
        expected=old.compute(morphology,physical,q,base,refs)
        actual=new.compute(morphology,physical,q,base,refs)
        for anchor in expected.anchor_poses_world:
            np.testing.assert_allclose(actual.anchor_poses_world[anchor][:3],
                expected.anchor_poses_world[anchor][:3],atol=1e-12,rtol=0)
            np.testing.assert_allclose(actual.anchor_jacobians[anchor],
                expected.anchor_jacobians[anchor],atol=1e-8,rtol=0)
        expected_com=_centroidal_pose(morphology,physical,q,
            expected.module_root_poses_world,builder)
        com = (*map(float, new._evaluate_batch(morphology=morphology, q_samples=[q],
            base_pose_world=base, references=())[-1][0]), *base[3:])
        reference_builder=SimpleNamespace(body_pose=lambda m,p,o: builder.build(m,p,o).body_pose_world)
        reference_com=_centroidal_pose(morphology,physical,q,expected.module_root_poses_world,reference_builder)
        assert expected_com==reference_com
        np.testing.assert_allclose(com,expected_com,atol=1e-12,rtol=0)
        restored=base_pose_for_centroidal_target(morphology,physical,q,
            com[:3],com[3:],kinematics=new)
        np.testing.assert_allclose(restored,base,atol=1e-12,rtol=0)
        values.append(com)
    assert values[0]==values[-1]
    assert values[0]!=values[1]


def test_fk_cache_keeps_exact_values_and_rechecks_new_models(grasp_carry_dict):
    from copy import deepcopy
    _,physical,context=_system(grasp_carry_dict)
    m=context.morphology_graph
    refs=resolve_mesh_backed_anchor_references(m,physical,[a.anchor_id for a in m.robot_anchors])
    ids=ordered_global_dock_joint_ids(m,physical)
    k=WholeStructureKinematics();base=(.4,.1,.7,0.,0.,0.,1.)
    q={j:0. for j in ids}
    first=k.compute(m,physical,q,base,refs)
    moved={j:.12 for j in ids}
    k.compute(m,physical,moved,base,refs)
    assert k.compute(m,physical,q,base,refs)==first
    changed=deepcopy(physical)
    changed.joints[0].origin_xyz=tuple(x+.001 for x in changed.joints[0].origin_xyz)
    assert k.compute(m,changed,q,base,refs)==WholeStructureKinematics().compute(m,changed,q,base,refs)


def test_pose_only_model_rechecks_in_place_model_and_observation_changes(grasp_carry_dict):
    _, physical, context = _system(grasp_carry_dict)
    builder = RigidBodyControlModelBuilder()
    morphology = context.morphology_graph
    observation = context.runtime_observation
    def check():
        assert builder.body_pose(morphology, physical, observation) == builder.build(
            morphology, physical, observation).body_pose_world
    check()
    physical.joints[0].origin_xyz = tuple(x + .002 for x in physical.joints[0].origin_xyz)
    check()
    physical.links[0].mass_kg *= 1.2
    physical.links[0].local_com = (.03, .02, .01)
    check()
    state = observation.module_states[0]
    state.pose_world = (state.pose_world[0] + .05, *state.pose_world[1:])
    check()
    if state.joint_positions:
        joint = next(iter(state.joint_positions))
        state.joint_positions[joint] += .13
    check()


def test_collision_reuse_rechecks_scene_state_margin_and_permissions(grasp_carry_dict):
    from amsrr.training.order9_posture_resolver import Order9PostureTrajectoryResolver
    from tests.unit.training.test_order9_articulated_teacher import _collision_object
    task, physical, context = _system(grasp_carry_dict)
    m = context.morphology_graph
    obj = _collision_object(task)
    reused = Order9PostureTrajectoryResolver(physical, collision_object=obj,
        require_native_solver=True).ik_solver
    anchors = tuple(a.anchor_id for a in m.robot_anchors)
    ids = ordered_global_dock_joint_ids(m, physical)
    # Compare reused robot geometry against an independently rebuilt kernel.
    for i, (shift, margin, allowed, exact) in enumerate([
            (0., .001, anchors, False), (.02, .001, (), False),
            (0., .005, anchors, False), (0., .001, anchors, True),
            (0., .001, (), True), (0., .001, anchors, False)]):
        fresh = Order9PostureTrajectoryResolver(physical, collision_object=obj,
            require_native_solver=True).ik_solver
        scene = dict(morphology=m,
            object_pose_world=(obj.initial_pose_world[0]+shift, *obj.initial_pose_world[1:]),
            object_size_m=obj.size_m, allowed_anchor_ids=allowed)
        body = (.1 if i == 2 else 0., 0., .7, 0., 0., 0., 1.)
        query = dict(morphology=m, centroidal_pose_world=body,
            joint_positions_rad={j: .1 if i == 4 else 0. for j in ids},
            exact=exact, margin_m=margin, ground_plane_z_m=0.)
        reused.set_collision_scene(**scene)
        fresh.set_collision_scene(**scene)
        if exact:
            # Proxy-only builds must reject exact-mesh requests, also after cache hits.
            for solver in (reused, fresh):
                with pytest.raises(RuntimeError, match="exact collision geometry was not loaded"):
                    solver.check_configuration(**query)
        else:
            assert reused.check_configuration(**query) == fresh.check_configuration(**query)
