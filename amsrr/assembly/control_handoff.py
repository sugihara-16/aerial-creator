from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from amsrr.assembly.construction_state import AssemblyStep, ConstructionState
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph


ControlHandoffMode = Literal["component_motion", "docking", "split_release", "safe_hold"]


@dataclass
class ControlHandoffRequest(SchemaBase):
    step_id: int
    control_mode: ControlHandoffMode
    active_module_ids: list[int]
    leader_module_id: int
    follower_module_id: int | None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def validate(self) -> None:
        if self.step_id < 0 or self.leader_module_id < 0:
            raise SchemaValidationError("ControlHandoffRequest ids must be non-negative")
        require_non_empty(self.control_mode, "ControlHandoffRequest.control_mode")


class ControlHandoffManager:
    """Interface shim between deterministic assembly steps and controllers."""

    def build_request(self, step: AssemblyStep, state: ConstructionState) -> ControlHandoffRequest:
        mode = _mode_for_step(step.step_type)
        active_modules = _active_modules_for_step(step, state)
        return ControlHandoffRequest(
            step_id=step.step_id,
            control_mode=mode,
            active_module_ids=active_modules,
            leader_module_id=step.leader_module_id,
            follower_module_id=step.follower_module_id,
            metadata={
                "step_type": step.step_type,
                "src_port_id": step.src_port_id if step.src_port_id is not None else -1,
                "dst_port_id": step.dst_port_id if step.dst_port_id is not None else -1,
            },
        )

    def build_assembly_control_request(
        self,
        step: AssemblyStep,
        state: ConstructionState,
        target_graph: MorphologyGraph,
    ):
        """Build the typed Order-5 component request for one attach sequence.

        The import is local so the legacy P3 handoff schema remains usable
        without creating an assembly-module import cycle.
        """

        from amsrr.assembly.assembly_control_bridge import (
            AssemblyComponentSpec,
            AssemblyControlRequest,
        )

        if step.step_type not in {
            "move_to_staging",
            "align_ports",
            "dock",
            "verify_attach",
        }:
            raise SchemaValidationError(
                "AssemblyControlRequest is only defined for attach-sequence steps"
            )
        if (
            step.follower_module_id is None
            or step.src_port_id is None
            or step.dst_port_id is None
        ):
            raise SchemaValidationError(
                "Attach-sequence AssemblyStep requires follower and both port ids"
            )
        leader_modules = _component_for_module(
            state,
            step.leader_module_id,
        )
        follower_modules = _component_for_module(
            state,
            step.follower_module_id,
        )
        if set(leader_modules) & set(follower_modules):
            raise SchemaValidationError(
                "Attach-sequence leader and follower must be separate components"
            )
        ports = {port.port_global_id: port for port in target_graph.ports}
        src_port = ports.get(step.src_port_id)
        dst_port = ports.get(step.dst_port_id)
        if src_port is None or dst_port is None:
            raise SchemaValidationError(
                "Attach-sequence AssemblyStep references a missing target port"
            )
        port_by_module = {
            src_port.module_id: src_port.port_global_id,
            dst_port.module_id: dst_port.port_global_id,
        }
        leader_port_id = port_by_module.get(step.leader_module_id)
        follower_port_id = port_by_module.get(step.follower_module_id)
        if leader_port_id is None or follower_port_id is None:
            raise SchemaValidationError(
                "Attach-sequence ports do not belong to the leader/follower endpoints"
            )
        return AssemblyControlRequest(
            step_id=step.step_id,
            leader=AssemblyComponentSpec(
                component_id=_component_id(leader_modules),
                module_ids=leader_modules,
            ),
            follower=AssemblyComponentSpec(
                component_id=_component_id(follower_modules),
                module_ids=follower_modules,
            ),
            leader_port_id=leader_port_id,
            follower_port_id=follower_port_id,
        )


def _mode_for_step(step_type: str) -> ControlHandoffMode:
    if step_type in {"move_to_staging", "align_ports"}:
        return "component_motion"
    if step_type in {"dock", "verify_attach"}:
        return "docking"
    if step_type == "detach":
        return "split_release"
    return "safe_hold"


def _active_modules_for_step(step: AssemblyStep, state: ConstructionState) -> list[int]:
    modules = {step.leader_module_id}
    if step.follower_module_id is not None:
        modules.add(step.follower_module_id)
    modules.update(module.module_id for module in state.control_graph.modules)
    return sorted(modules)


def _component_for_module(state: ConstructionState, module_id: int) -> list[int]:
    matches = [
        sorted(component)
        for component in state.attached_components
        if module_id in component
    ]
    if len(matches) != 1:
        raise SchemaValidationError(
            f"ConstructionState does not uniquely locate module {module_id} in an attached component"
        )
    return matches[0]


def _component_id(module_ids: list[int]) -> str:
    return "component:" + "-".join(str(module_id) for module_id in sorted(module_ids))
