from __future__ import annotations

from amsrr.assembly import (
    ControlHandoffManager,
    GraphEditAssemblyPlanner,
    initial_construction_state,
    mark_edge_attached,
)
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.graph import build_minimal_design_output
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.task_spec import TaskSpec


def _target_graph(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = build_minimal_design_output(task, irg, physical_model)
    return design.target_morphology


def test_initial_construction_state_contains_base_only(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    state = initial_construction_state(target_graph)

    assert [module.module_id for module in state.physical_graph.modules] == [target_graph.base_module_id]
    assert state.unattached_modules == [1, 2]
    assert state.attached_components == [[0], [1], [2]]
    assert state.physical_graph.dock_edges == []
    assert all(not port.occupied for port in state.physical_graph.ports)


def test_graph_edit_planner_builds_deterministic_attach_sequence(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    plan = GraphEditAssemblyPlanner().build_plan(target_graph)

    assert plan.target_graph_id == target_graph.graph_id
    assert len(plan.steps) == 4 * len(target_graph.dock_edges)
    assert [step.step_id for step in plan.steps] == list(range(len(plan.steps)))
    assert [step.step_type for step in plan.steps[:4]] == [
        "move_to_staging",
        "align_ports",
        "dock",
        "verify_attach",
    ]
    assert plan.steps[0].leader_module_id == 0
    assert plan.steps[0].follower_module_id == 1
    assert plan.steps[1].target_relative_pose == target_graph.dock_edges[0].relative_pose_src_to_dst
    assert plan.steps[3].success_conditions[0]["type"] == "edge_attached"
    assert plan.estimated_duration_s == sum(step.timeout_s for step in plan.steps)
    assert type(plan).from_json(plan.to_json()).to_dict() == plan.to_dict()


def test_graph_edit_planner_resumes_from_construction_state(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    state = initial_construction_state(target_graph)
    state = mark_edge_attached(state, target_graph, target_graph.dock_edges[0].edge_id)

    next_step = GraphEditAssemblyPlanner().next_step(target_graph, construction_state=state)

    assert next_step is not None
    assert next_step.step_type == "move_to_staging"
    assert next_step.leader_module_id == 1
    assert next_step.follower_module_id == 2
    assert state.unattached_modules == [2]
    assert state.attached_components == [[0, 1], [2]]
    assert state.docking_attempts == {"0": 1}


def test_control_handoff_request_for_docking_step(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    state = initial_construction_state(target_graph)
    plan = GraphEditAssemblyPlanner().build_plan(target_graph)
    dock_step = plan.steps[2]

    request = ControlHandoffManager().build_request(dock_step, state)

    assert request.control_mode == "docking"
    assert request.leader_module_id == dock_step.leader_module_id
    assert request.follower_module_id == dock_step.follower_module_id
    assert set(request.active_module_ids) == {0, 1}
    assert request.metadata["step_type"] == "dock"


def test_control_handoff_builds_component_scoped_order5_request(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    state = initial_construction_state(target_graph)
    step = GraphEditAssemblyPlanner().build_plan(target_graph).steps[0]

    request = ControlHandoffManager().build_assembly_control_request(
        step,
        state,
        target_graph,
    )

    assert request.leader.module_ids == [0]
    assert request.follower.module_ids == [1]
    assert request.leader.component_id == "component:0"
    assert request.follower.component_id == "component:1"
    leader_port = next(
        port for port in target_graph.ports if port.port_global_id == request.leader_port_id
    )
    follower_port = next(
        port for port in target_graph.ports if port.port_global_id == request.follower_port_id
    )
    assert leader_port.module_id == step.leader_module_id
    assert follower_port.module_id == step.follower_module_id
