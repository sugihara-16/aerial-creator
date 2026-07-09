from __future__ import annotations

from dataclasses import dataclass, field

from amsrr.controllers.actuator_mapping import build_actuator_mapping
from amsrr.controllers.controller_base import ControllerContext
from amsrr.controllers.isaac_controller_bridge import IsaacActuatorTargetRecord, IsaacControllerBridge
from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
from amsrr.robot_model.physical_model_builder import build_module_capability_token
from amsrr.schemas.morphology import ControlGroup, ModuleNode, MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ControllerCommand, ControllerStatus, InteractionKnot, PolicyCommand
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState


@dataclass
class ControllerCommandSmokeBundle:
    morphology_graph: MorphologyGraph
    runtime_observation: RuntimeObservation
    controller_command: ControllerCommand
    actuator_target_record: IsaacActuatorTargetRecord
    metrics: dict[str, float] = field(default_factory=dict)


def build_single_module_controller_command_smoke(
    physical_model: PhysicalModel,
    *,
    graph_id: str = "single-module-controller-command-smoke",
    time_s: float = 0.0,
    command_index: int = 0,
    control_dt_s: float = 0.005,
    pose_world: tuple[float, float, float, float, float, float, float] = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
    twist_world: list[float] | None = None,
    joint_positions: dict[str, float] | None = None,
    joint_velocities: dict[str, float] | None = None,
    previous_command: ControllerCommand | None = None,
) -> ControllerCommandSmokeBundle:
    morphology_graph = build_single_module_morphology(physical_model, graph_id=graph_id)
    runtime_observation = build_runtime_observation(
        morphology_graph,
        time_s=time_s,
        pose_world=pose_world,
        twist_world=twist_world or [0.0] * 6,
        joint_positions=joint_positions or {},
        joint_velocities=joint_velocities or {},
    )
    active_knot = InteractionKnot(t_rel_s=time_s, contact_assignments=[])
    controller = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode="rigid_body_qp",
            control_dt_s=control_dt_s,
        )
    )
    controller_command = controller.compute(
        ControllerContext(
            runtime_observation=runtime_observation,
            morphology_graph=morphology_graph,
            physical_model=physical_model,
            active_knot=active_knot,
            policy_command=PolicyCommand(),
            previous_command=previous_command,
            control_dt_s=control_dt_s,
        )
    )
    bridged_command = bridge_supported_controller_command(controller_command)
    actuator_mapping = build_actuator_mapping(morphology_graph, physical_model)
    actuator_target_record = IsaacControllerBridge().convert(
        bridged_command,
        actuator_mapping,
        time_s=time_s,
        command_index=command_index,
    )
    return ControllerCommandSmokeBundle(
        morphology_graph=morphology_graph,
        runtime_observation=runtime_observation,
        controller_command=bridged_command,
        actuator_target_record=actuator_target_record,
        metrics={
            "raw_joint_torque_command_count": float(len(controller_command.joint_torque_commands)),
            "controller_rotor_count": float(len(bridged_command.rotor_thrusts_n)),
            "controller_vectoring_target_count": float(len(bridged_command.vectoring_joint_targets)),
            "bridge_target_count": actuator_target_record.metrics["actuator_target_count"],
            "bridge_missing_actuator_count": actuator_target_record.metrics["missing_actuator_count"],
            "bridge_unsupported_actuator_count": actuator_target_record.metrics["unsupported_actuator_count"],
            "controller_qp_feasible": 1.0 if bridged_command.controller_status.qp_feasible else 0.0,
        },
    )


def build_single_module_morphology(
    physical_model: PhysicalModel,
    *,
    graph_id: str = "single-module-controller-command-smoke",
) -> MorphologyGraph:
    capability = build_module_capability_token(physical_model)
    return MorphologyGraph(
        graph_id=graph_id,
        modules=[
            ModuleNode(
                module_id=0,
                module_type="holon",
                pose_in_design_frame=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                role_id="base",
                is_base=True,
                capability_token=capability,
            )
        ],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[ControlGroup(group_id="all", module_ids=[0], role="whole_body")],
        base_module_id=0,
        is_closed_loop=False,
    )


def build_runtime_observation(
    morphology_graph: MorphologyGraph,
    *,
    time_s: float,
    pose_world: tuple[float, float, float, float, float, float, float],
    twist_world: list[float],
    joint_positions: dict[str, float],
    joint_velocities: dict[str, float],
) -> RuntimeObservation:
    return RuntimeObservation(
        time_s=time_s,
        morphology_graph=morphology_graph,
        module_states=[
            ModuleRuntimeState(
                module_id=morphology_graph.base_module_id,
                pose_world=pose_world,
                twist_world=twist_world,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
            )
        ],
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )


def bridge_supported_controller_command(controller_command: ControllerCommand) -> ControllerCommand:
    vectoring_targets = dict(controller_command.vectoring_joint_targets)
    for command_key in list(vectoring_targets):
        if command_key.startswith("module_"):
            continue
        global_key = f"module_0:{command_key}"
        if global_key in vectoring_targets:
            del vectoring_targets[command_key]
    return ControllerCommand(
        rotor_thrusts_n=dict(controller_command.rotor_thrusts_n),
        vectoring_joint_targets=vectoring_targets,
        joint_torque_commands={},
        dock_mechanism_commands=dict(controller_command.dock_mechanism_commands),
        controller_status=controller_command.controller_status,
    )
