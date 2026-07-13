from __future__ import annotations

import sys
import types

import pytest

from amsrr.geometry.pose_math import FACE_TO_FACE_DOCK_RELATION, compose_pose
from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.simulation.dynamic_dock_constraint import (
    build_dynamic_dock_constraint_spec,
    connect_frame_pose_in_parent_body,
    constraint_residual,
    filter_selected_body_pair,
    fixed_joint_identity_failures,
    selected_body_pair_filter_state,
    selected_body_pair_filter_failures,
)


def _graph_and_model():
    model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    graph = RandomConnectedMorphologyDistribution(model).sample(seed=2, module_count=2)
    return graph, model


def test_constraint_spec_uses_dock_body_local_connect_frames_and_face_relation() -> None:
    graph, model = _graph_and_model()
    edge = graph.dock_edges[0]
    leader_port = next(port for port in graph.ports if port.port_global_id == edge.src_port_id)
    follower_port = next(port for port in graph.ports if port.port_global_id == edge.dst_port_id)

    spec = build_dynamic_dock_constraint_spec(
        graph,
        model,
        edge_id=edge.edge_id,
        leader_module_id=edge.src_module_id,
        follower_module_id=edge.dst_module_id,
        leader_body_path="/World/Assembly/Leader/pitch_dock_mech1",
        follower_body_path="/World/Assembly/Follower/yaw_dock_mech1",
    )

    assert spec.leader_body_local_connect_pose == connect_frame_pose_in_parent_body(leader_port, model)
    assert spec.follower_body_local_constraint_pose == compose_pose(
        connect_frame_pose_in_parent_body(follower_port, model),
        FACE_TO_FACE_DOCK_RELATION,
    )
    assert spec.leader_body_local_connect_pose[:3] in {
        (0.045, 0.0, 0.0),
        (0.115, 0.0, 0.0),
    }
    assert type(spec).from_json(spec.to_json()).to_dict() == spec.to_dict()


def test_constraint_residual_is_zero_only_for_face_to_face_frames() -> None:
    leader = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    follower = compose_pose(leader, FACE_TO_FACE_DOCK_RELATION)

    residual = constraint_residual(
        leader,
        follower,
        leader_connect_twist_world=[0.0] * 6,
        follower_connect_twist_world=[0.0] * 6,
    )

    assert residual.position_error_m == pytest.approx(0.0)
    assert residual.attitude_error_rad == pytest.approx(0.0)
    assert residual.relative_linear_speed_mps == pytest.approx(0.0)
    assert residual.relative_angular_speed_radps == pytest.approx(0.0)

    wrong = constraint_residual(
        leader,
        leader,
        leader_connect_twist_world=[0.0] * 6,
        follower_connect_twist_world=[0.1, 0.0, 0.0, 0.0, 0.0, 0.2],
    )
    assert wrong.attitude_error_rad == pytest.approx(3.141592653589793)
    assert wrong.relative_linear_speed_mps == pytest.approx(0.1)
    assert wrong.relative_angular_speed_radps == pytest.approx(0.2)


class _FakeAttribute:
    def __init__(self, value):
        self._value = value

    def Get(self):
        return self._value


class _FakeRelationship:
    def __init__(self, *paths: str):
        self._targets = [types.SimpleNamespace(pathString=path) for path in paths]

    def GetTargets(self):
        return self._targets

    def AddTarget(self, path):
        path_string = getattr(path, "pathString", str(path))
        if path_string not in {target.pathString for target in self._targets}:
            self._targets.append(types.SimpleNamespace(pathString=path_string))

    def RemoveTarget(self, path):
        path_string = getattr(path, "pathString", str(path))
        self._targets = [
            target for target in self._targets if target.pathString != path_string
        ]


class _FakeQuaternion:
    def __init__(self, pose):
        self._imaginary = pose[3:6]
        self._real = pose[6]

    def GetImaginary(self):
        return self._imaginary

    def GetReal(self):
        return self._real


class _FakeFixedJoint:
    def __new__(cls, prim):
        return prim.joint


class _FakeJoint:
    def __init__(self, spec, *, enabled):
        self._enabled = _FakeAttribute(enabled)
        self._body0 = _FakeRelationship(spec.leader_body_path)
        self._body1 = _FakeRelationship(spec.follower_body_path)
        self._exclude = _FakeAttribute(True)
        self._collision = _FakeAttribute(True)
        self._positions = (
            _FakeAttribute(spec.leader_body_local_connect_pose[:3]),
            _FakeAttribute(spec.follower_body_local_constraint_pose[:3]),
        )
        self._rotations = (
            _FakeAttribute(_FakeQuaternion(spec.leader_body_local_connect_pose)),
            _FakeAttribute(_FakeQuaternion(spec.follower_body_local_constraint_pose)),
        )

    def GetJointEnabledAttr(self):
        return self._enabled

    def GetBody0Rel(self):
        return self._body0

    def GetBody1Rel(self):
        return self._body1

    def GetExcludeFromArticulationAttr(self):
        return self._exclude

    def GetCollisionEnabledAttr(self):
        return self._collision

    def GetLocalPos0Attr(self):
        return self._positions[0]

    def GetLocalPos1Attr(self):
        return self._positions[1]

    def GetLocalRot0Attr(self):
        return self._rotations[0]

    def GetLocalRot1Attr(self):
        return self._rotations[1]


class _FakeRigidBodyAPI:
    pass


class _FakeFilteredPairsAPI:
    def __init__(self, prim):
        self._prim = prim

    @classmethod
    def Apply(cls, prim):
        return cls(prim)

    def GetFilteredPairsRel(self):
        return self._prim.filtered_relationship

    def CreateFilteredPairsRel(self):
        return self._prim.filtered_relationship


class _FakePrim:
    def __init__(
        self,
        joint=None,
        *,
        path: str | None = None,
        rigid_body: bool = True,
        filtered_targets: tuple[str, ...] = (),
    ):
        self.joint = joint
        self._path = path
        self._rigid_body = rigid_body
        self.filtered_relationship = _FakeRelationship(*filtered_targets)

    def IsValid(self):
        return self.joint is not None

    def IsA(self, schema_type):
        return self.IsValid() and schema_type is _FakeFixedJoint

    def HasAPI(self, schema_type):
        return (
            self.IsValid()
            and schema_type is _FakeRigidBodyAPI
            and self._rigid_body
        )

    def GetPath(self):
        return self._path


class _FakeStage:
    def __init__(self, path: str, prim: _FakePrim | None):
        self._path = path
        self._prim = prim

    def GetPrimAtPath(self, path):
        if path == self._path and self._prim is not None:
            return self._prim
        return _FakePrim()


def _install_fake_pxr(monkeypatch: pytest.MonkeyPatch) -> None:
    pxr = types.ModuleType("pxr")
    pxr.Sdf = types.SimpleNamespace(Path=lambda value: value)
    pxr.UsdPhysics = types.SimpleNamespace(
        FilteredPairsAPI=_FakeFilteredPairsAPI,
        FixedJoint=_FakeFixedJoint,
        RigidBodyAPI=_FakeRigidBodyAPI,
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    physx = types.ModuleType("omni.physx")
    physx.get_physx_simulation_interface = lambda: types.SimpleNamespace(
        flush_changes=lambda: None
    )
    omni = types.ModuleType("omni")
    omni.physx = physx
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.physx", physx)


def _constraint_spec():
    graph, model = _graph_and_model()
    edge = graph.dock_edges[0]
    return build_dynamic_dock_constraint_spec(
        graph,
        model,
        edge_id=edge.edge_id,
        leader_module_id=edge.src_module_id,
        follower_module_id=edge.dst_module_id,
        leader_body_path="/World/Assembly/Leader/pitch_dock_mech1",
        follower_body_path="/World/Assembly/Follower/yaw_dock_mech1",
    )


def _selected_pair_stage(
    spec,
    *,
    leader_targets: tuple[str, ...] = (),
    follower_targets: tuple[str, ...] = (),
    leader_rigid_body: bool = True,
    follower_rigid_body: bool = True,
):
    leader = _FakePrim(
        object(),
        path=spec.leader_body_path,
        rigid_body=leader_rigid_body,
        filtered_targets=leader_targets,
    )
    follower = _FakePrim(
        object(),
        path=spec.follower_body_path,
        rigid_body=follower_rigid_body,
        filtered_targets=follower_targets,
    )

    class _PairStage:
        def GetPrimAtPath(self, path):
            return {
                spec.leader_body_path: leader,
                spec.follower_body_path: follower,
            }.get(path, _FakePrim())

    return _PairStage(), leader, follower


def test_constraint_identity_requires_enabled_joint_by_default(monkeypatch) -> None:
    _install_fake_pxr(monkeypatch)
    spec = _constraint_spec()

    enabled_stage = _FakeStage(spec.prim_path, _FakePrim(_FakeJoint(spec, enabled=True)))
    disabled_stage = _FakeStage(spec.prim_path, _FakePrim(_FakeJoint(spec, enabled=False)))

    assert fixed_joint_identity_failures(enabled_stage, spec) == []
    assert fixed_joint_identity_failures(disabled_stage, spec) == [
        "constraint_joint_enabled_mismatch"
    ]


def test_constraint_identity_can_verify_preauthored_disabled_state(monkeypatch) -> None:
    _install_fake_pxr(monkeypatch)
    spec = _constraint_spec()
    disabled_stage = _FakeStage(spec.prim_path, _FakePrim(_FakeJoint(spec, enabled=False)))

    assert fixed_joint_identity_failures(
        disabled_stage,
        spec,
        expected_enabled=False,
    ) == []


def test_constraint_identity_reports_missing_local_frame_without_throwing(monkeypatch) -> None:
    _install_fake_pxr(monkeypatch)
    spec = _constraint_spec()
    joint = _FakeJoint(spec, enabled=True)
    joint._positions = (
        _FakeAttribute(None),
        joint._positions[1],
    )
    stage = _FakeStage(spec.prim_path, _FakePrim(joint))

    assert "constraint_local_frame_0_missing" in fixed_joint_identity_failures(
        stage,
        spec,
    )


def test_constraint_identity_rejects_removed_joint(monkeypatch) -> None:
    _install_fake_pxr(monkeypatch)
    spec = _constraint_spec()
    removed_stage = _FakeStage(spec.prim_path, None)

    assert fixed_joint_identity_failures(removed_stage, spec) == [
        "constraint_prim_missing"
    ]


def test_selected_body_filter_identity_checks_exact_target(monkeypatch) -> None:
    _install_fake_pxr(monkeypatch)
    spec = _constraint_spec()
    stage, leader, _follower = _selected_pair_stage(
        spec,
        leader_targets=(spec.follower_body_path,),
    )
    assert selected_body_pair_filter_failures(
        stage,
        spec,
        expected_filtered=True,
    ) == []
    assert selected_body_pair_filter_failures(
        stage,
        spec,
        expected_filtered=False,
    ) == ["selected_filter_target_state_mismatch"]

    leader.filtered_relationship = _FakeRelationship()
    assert selected_body_pair_filter_failures(
        stage,
        spec,
        expected_filtered=False,
    ) == []


def test_selected_body_pair_filter_adds_only_owned_target_and_preserves_unrelated(
    monkeypatch,
) -> None:
    _install_fake_pxr(monkeypatch)
    spec = _constraint_spec()
    unrelated_leader_target = "/World/Unrelated/LeaderTarget"
    unrelated_follower_target = "/World/Unrelated/FollowerTarget"
    stage, _leader, _follower = _selected_pair_stage(
        spec,
        leader_targets=(unrelated_leader_target,),
        follower_targets=(unrelated_follower_target,),
    )

    before = selected_body_pair_filter_state(stage, spec)
    delta = filter_selected_body_pair(stage, spec)
    after = selected_body_pair_filter_state(stage, spec)

    assert before["leader_targets"] == [unrelated_leader_target]
    assert before["follower_targets"] == [unrelated_follower_target]
    assert delta["added_leader_targets"] == [spec.follower_body_path]
    assert delta["removed_leader_targets"] == []
    assert delta["added_follower_targets"] == []
    assert delta["removed_follower_targets"] == []
    assert after["leader_targets"] == sorted(
        [unrelated_leader_target, spec.follower_body_path]
    )
    assert after["follower_targets"] == [unrelated_follower_target]


@pytest.mark.parametrize("preexisting_direction", ["forward", "reverse"])
def test_selected_body_pair_filter_rejects_preexisting_pair_in_either_direction(
    monkeypatch,
    preexisting_direction: str,
) -> None:
    _install_fake_pxr(monkeypatch)
    spec = _constraint_spec()
    leader_targets = (
        (spec.follower_body_path,) if preexisting_direction == "forward" else ()
    )
    follower_targets = (
        (spec.leader_body_path,) if preexisting_direction == "reverse" else ()
    )
    stage, _leader, _follower = _selected_pair_stage(
        spec,
        leader_targets=leader_targets,
        follower_targets=follower_targets,
    )
    before = selected_body_pair_filter_state(stage, spec)

    with pytest.raises(
        RuntimeError,
        match="selected dock body pair was already collision-filtered",
    ):
        filter_selected_body_pair(stage, spec)

    assert selected_body_pair_filter_state(stage, spec) == before


@pytest.mark.parametrize(
    ("leader_rigid_body", "follower_rigid_body"),
    [(False, True), (True, False)],
)
def test_selected_body_pair_filter_rejects_non_rigid_selected_body(
    monkeypatch,
    leader_rigid_body: bool,
    follower_rigid_body: bool,
) -> None:
    _install_fake_pxr(monkeypatch)
    spec = _constraint_spec()
    unrelated_target = "/World/Unrelated/Target"
    stage, _leader, _follower = _selected_pair_stage(
        spec,
        leader_targets=(unrelated_target,),
        leader_rigid_body=leader_rigid_body,
        follower_rigid_body=follower_rigid_body,
    )
    before = selected_body_pair_filter_state(stage, spec)

    with pytest.raises(
        RuntimeError,
        match="selected dock body prim is not a rigid body",
    ):
        filter_selected_body_pair(stage, spec)

    assert selected_body_pair_filter_state(stage, spec) == before
