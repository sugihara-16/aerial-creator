from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from amsrr.assembly.construction_state import AssemblyStep, ConstructionState
from amsrr.schemas.common import SchemaBase, require_non_empty


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
            from amsrr.schemas.common import SchemaValidationError

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
