from __future__ import annotations

import math

import pytest

from amsrr.assembly.assembly_control_bridge import (
    AssemblyComponentObservation,
    AssemblyControlBridge,
    AssemblyControlBridgeConfig,
    AssemblyControlObservation,
)
from amsrr.assembly.closed_loop_executor import (
    ClosedLoopAssemblyExecutor,
    _angular_velocity_toward_pose,
    _attitude_error,
    _bounded_pose_step,
    _linear_velocity_toward_pose,
    _position_error,
)
from amsrr.assembly.construction_state import initial_construction_state
from amsrr.assembly.graph_edit_planner import GraphEditAssemblyPlanner
from amsrr.geometry.pose_math import compose_pose
from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config


class _FakeAssemblyRuntime:
    control_dt_s = 0.05

    def __init__(self, graph, physical_model) -> None:
        edge = graph.dock_edges[0]
        self.graph = graph
        self.physical_model = physical_model
        self.leader_port = next(port for port in graph.ports if port.port_global_id == edge.src_port_id)
        self.follower_port = next(port for port in graph.ports if port.port_global_id == edge.dst_port_id)
        self.time_s = 0.0
        self.leader_body = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.follower_body = (0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.constraint_present = False
        self.constraint_verified = False
        self.last_commands = None

    def observe(self):
        leader_connect = compose_pose(self.leader_body, self.leader_port.local_pose)
        follower_connect = compose_pose(self.follower_body, self.follower_port.local_pose)
        relative_x = follower_connect[0] - leader_connect[0]
        contact = abs(relative_x) <= 0.003
        return AssemblyControlObservation(
            time_s=self.time_s,
            components=[
                AssemblyComponentObservation(
                    component_id="component:0",
                    module_ids=[0],
                    body_pose_world=self.leader_body,
                    selected_connect_pose_world=leader_connect,
                    selected_connect_linear_velocity_world=(0.0, 0.0, 0.0),
                    selected_connect_angular_velocity_world=(0.0, 0.0, 0.0),
                    qp_feasible=True,
                ),
                AssemblyComponentObservation(
                    component_id="component:1",
                    module_ids=[1],
                    body_pose_world=self.follower_body,
                    selected_connect_pose_world=follower_connect,
                    selected_connect_linear_velocity_world=(0.0, 0.0, 0.0),
                    selected_connect_angular_velocity_world=(0.0, 0.0, 0.0),
                    qp_feasible=True,
                ),
            ],
            selected_pair_contact=contact,
            selected_pair_contact_evidence_valid=contact,
            selected_pair_contact_force_n=1.0 if contact else 0.0,
            selected_pair_penetration_m=0.0,
            constraint_present=self.constraint_present,
            constraint_verified=self.constraint_verified,
        )

    def apply_and_step(self, commands) -> None:
        self.last_commands = commands
        follower = next(target for target in commands.component_targets if target.role == "follower")
        target = follower.policy_command.desired_body_pose
        if target is not None:
            self.follower_body = target
        if commands.constraint_intent.action == "create":
            self.constraint_present = True
        elif commands.constraint_intent.action == "verify":
            self.constraint_verified = True
        self.time_s += self.control_dt_s

    def is_component_pose_collision_free(self, component_id, pose_world) -> bool:
        return component_id == "component:1" and pose_world[2] > 0.5


def test_closed_loop_executor_spans_the_legacy_four_step_sequence() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    graph = RandomConnectedMorphologyDistribution(physical_model).sample(seed=2, module_count=2)
    edge = graph.dock_edges[0]
    # Normalize this deterministic fake so src is leader module 0.
    assert edge.src_module_id == 0
    runtime = _FakeAssemblyRuntime(graph, physical_model)
    bridge = AssemblyControlBridge(
        graph,
        {0: physical_model, 1: physical_model},
        config=AssemblyControlBridgeConfig(
            staging_offset_m=0.15,
            staging_axial_tolerance_m=0.02,
            transverse_tolerance_m=0.02,
            attitude_tolerance_rad=0.10,
            relative_linear_speed_tolerance_mps=0.10,
            relative_angular_speed_tolerance_radps=0.10,
            prealign_dwell_s=0.05,
            approach_speed_mps=1.0,
            fix_axial_tolerance_m=0.003,
            selected_contact_dwell_s=0.05,
            max_selected_contact_force_n=5.0,
            max_selected_contact_penetration_m=0.001,
            step_timeout_s=5.0,
        ),
    )
    executor = ClosedLoopAssemblyExecutor(
        target_graph=graph,
        bridge=bridge,
        runtime=runtime,
    )
    state = initial_construction_state(graph)
    steps = GraphEditAssemblyPlanner().build_plan(graph).steps

    results = [executor.execute_step(step, state) for step in steps]

    assert all(result.success for result in results)
    assert [output.progress.phase for output in executor.trace if output.progress.completed] == ["verify"]
    assert runtime.constraint_present is True
    assert runtime.constraint_verified is True
    assert executor.trace[-1].progress.completed is True


def test_closed_loop_executor_fails_closed_when_collision_oracle_rejects_path() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    graph = RandomConnectedMorphologyDistribution(physical_model).sample(seed=2, module_count=2)
    runtime = _FakeAssemblyRuntime(graph, physical_model)
    runtime.is_component_pose_collision_free = lambda _component_id, _pose: False
    executor = ClosedLoopAssemblyExecutor(
        target_graph=graph,
        bridge=AssemblyControlBridge(graph, {0: physical_model, 1: physical_model}),
        runtime=runtime,
    )
    state = initial_construction_state(graph)
    step = GraphEditAssemblyPlanner().build_plan(graph).steps[0]

    result = executor.execute_step(step, state)

    assert result.success is False
    assert result.message is not None
    assert "staging_motion_plan_failed" in result.message


def test_staging_reference_is_pose_and_twist_bounded() -> None:
    current = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    target = (
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        math.sin(math.pi / 4.0),
        math.cos(math.pi / 4.0),
    )

    bounded = _bounded_pose_step(
        current,
        target,
        max_translation_step_m=0.03,
        max_angular_step_rad=0.05,
    )
    linear = _linear_velocity_toward_pose(current, target, max_speed_mps=0.10)
    angular = _angular_velocity_toward_pose(
        current,
        target,
        max_speed_radps=0.20,
    )

    assert _position_error(current, bounded) <= 0.03 + 1.0e-9
    assert _attitude_error(current, bounded) <= 0.05 + 1.0e-9
    assert math.sqrt(sum(value * value for value in linear)) == pytest.approx(0.10)
    assert math.sqrt(sum(value * value for value in angular)) == pytest.approx(0.20)
