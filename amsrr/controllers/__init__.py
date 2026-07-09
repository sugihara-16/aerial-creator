"""Controller-side interface helpers for A-MSRR."""

from amsrr.controllers.actuator_mapping import (
    ActuatorChannel,
    ActuatorMapping,
    ActuatorMappingBuilder,
    build_actuator_mapping,
)
from amsrr.controllers.controller_base import ControllerBase, ControllerContext
from amsrr.controllers.isaac_controller_bridge import (
    IsaacActuatorTarget,
    IsaacActuatorTargetRecord,
    IsaacControllerBridge,
    IsaacControllerBridgeConfig,
    actuator_target_record_to_dict,
)
from amsrr.controllers.policy_command_builder import DesiredBiasReferences, PolicyCommandBiasBuilder
from amsrr.controllers.qp_allocator_interface import (
    BoundedVerticalRotorAllocator,
    QPAllocationProblem,
    QPAllocationResult,
    QPAllocatorInterface,
    QP_INFEASIBLE_CODE,
    QP_RIGID_BODY_MODEL_REQUIRED_CODE,
    QP_SOLVER_UNAVAILABLE_CODE,
    QP_THRUST_CLIPPED_CODE,
    QP_UNSUPPORTED_WRENCH_CODE,
    QP_VECTORING_CLIPPED_CODE,
    RotorAllocationSpec,
    VirtualThrustQPAllocator,
)
from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
from amsrr.controllers.rigid_body_model import (
    RigidBodyControlModel,
    RigidBodyControlModelBuilder,
    RotorControlElement,
)

__all__ = [
    "ActuatorChannel",
    "ActuatorMapping",
    "ActuatorMappingBuilder",
    "BoundedVerticalRotorAllocator",
    "ControllerBase",
    "ControllerContext",
    "DesiredBiasReferences",
    "IsaacActuatorTarget",
    "IsaacActuatorTargetRecord",
    "IsaacControllerBridge",
    "IsaacControllerBridgeConfig",
    "PolicyCommandBiasBuilder",
    "QPAllocationProblem",
    "QPAllocationResult",
    "QPAllocatorInterface",
    "QPIDController",
    "QPIDControllerConfig",
    "QP_INFEASIBLE_CODE",
    "QP_RIGID_BODY_MODEL_REQUIRED_CODE",
    "QP_SOLVER_UNAVAILABLE_CODE",
    "QP_THRUST_CLIPPED_CODE",
    "QP_UNSUPPORTED_WRENCH_CODE",
    "QP_VECTORING_CLIPPED_CODE",
    "RigidBodyControlModel",
    "RigidBodyControlModelBuilder",
    "RotorAllocationSpec",
    "RotorControlElement",
    "VirtualThrustQPAllocator",
    "actuator_target_record_to_dict",
    "build_actuator_mapping",
]
