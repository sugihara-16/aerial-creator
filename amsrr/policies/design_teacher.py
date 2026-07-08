from __future__ import annotations

from dataclasses import dataclass, replace

from amsrr.morphology.graph import build_minimal_design_output
from amsrr.policies.design_candidate_generator import DesignCandidateGenerator, DesignCandidateStep
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.schemas.common import ContactMode, SchemaValidationError, StrEnum
from amsrr.schemas.irg import IRGNodeType
from amsrr.schemas.morphology import DesignOutput
from amsrr.schemas.task_spec import TaskType


class DesignTeacherVariant(StrEnum):
    CHAIN_GRASP = "chain_grasp"
    SYMMETRIC_TWO_ANCHOR_GRASP = "symmetric_two_anchor_grasp"
    TRI_ANCHOR_SUPPORT_GRASP = "tri_anchor_support_grasp"
    CENTRAL_BASE_PLUS_TWO_GRASP_ARMS = "central_base_plus_two_grasp_arms"
    PERCH_ANCHOR_FRAME = "perch_anchor_frame"
    VALVE_TORQUE_ARM = "valve_torque_arm"
    SUPPORT_SHIFT_FRAME = "support_shift_frame"


VARIANT_ORDER: tuple[DesignTeacherVariant, ...] = (
    DesignTeacherVariant.CHAIN_GRASP,
    DesignTeacherVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    DesignTeacherVariant.TRI_ANCHOR_SUPPORT_GRASP,
    DesignTeacherVariant.CENTRAL_BASE_PLUS_TWO_GRASP_ARMS,
    DesignTeacherVariant.PERCH_ANCHOR_FRAME,
    DesignTeacherVariant.VALVE_TORQUE_ARM,
    DesignTeacherVariant.SUPPORT_SHIFT_FRAME,
)


@dataclass(frozen=True)
class DesignTeacherExample:
    variant: DesignTeacherVariant
    design_output: DesignOutput
    candidate_trace: list[DesignCandidateStep]


class DeterministicDesignTeacher:
    """Demonstration morphology teacher for π_D bootstrapping.

    The current P1 implementation uses the existing minimal connected-tree
    builder and labels its output with a deterministic teacher variant. It is a
    bootstrap action-sequence provider, not a learned design policy.
    """

    def __init__(self, candidate_generator: DesignCandidateGenerator | None = None) -> None:
        self._candidate_generator = candidate_generator or DesignCandidateGenerator()

    def generate(
        self,
        context: DesignPolicyContext,
        *,
        variant: DesignTeacherVariant | str | None = None,
    ) -> DesignTeacherExample:
        selected_variant = DesignTeacherVariant(variant) if variant is not None else self.select_variant(context)
        design_output = build_minimal_design_output(context.task_spec, context.irg, context.physical_model)
        design_output = self._annotate_design_output(design_output, selected_variant, context)
        candidate_trace = self._candidate_generator.build_teacher_trace(design_output)
        return DesignTeacherExample(
            variant=selected_variant,
            design_output=design_output,
            candidate_trace=candidate_trace,
        )

    def select_variant(self, context: DesignPolicyContext) -> DesignTeacherVariant:
        task_type = context.task_spec.task_type
        if task_type == TaskType.OBJECT_GRASP_CARRY:
            return self._select_grasp_carry_variant(context)
        if task_type == TaskType.VALVE_OPERATION:
            return DesignTeacherVariant.VALVE_TORQUE_ARM
        if task_type == TaskType.PERCHING_MANIPULATION:
            return DesignTeacherVariant.PERCH_ANCHOR_FRAME
        if task_type == TaskType.CONTACT_MEDIATED_LOCOMOTION:
            return DesignTeacherVariant.SUPPORT_SHIFT_FRAME
        if task_type == TaskType.FREE_FLIGHT_NAVIGATION:
            return DesignTeacherVariant.CHAIN_GRASP
        raise SchemaValidationError(f"Unsupported task_type for design teacher: {task_type!r}")

    @staticmethod
    def _select_grasp_carry_variant(context: DesignPolicyContext) -> DesignTeacherVariant:
        required_grasp_min_count = 0
        has_optional_support = False
        for node in context.irg.nodes:
            if node.node_type != IRGNodeType.CONTACT_SLOT:
                continue
            mode = ContactMode(node.feature.get("contact_mode"))
            required = bool(node.feature.get("required", True))
            if mode == ContactMode.GRASP and required:
                required_grasp_min_count += int(node.feature.get("min_count_group", 1))
            if mode == ContactMode.SUPPORT and not required:
                has_optional_support = True
        if has_optional_support and context.task_spec.robot_constraints.max_modules >= 3:
            return DesignTeacherVariant.TRI_ANCHOR_SUPPORT_GRASP
        if required_grasp_min_count >= 2:
            return DesignTeacherVariant.SYMMETRIC_TWO_ANCHOR_GRASP
        return DesignTeacherVariant.CHAIN_GRASP

    @staticmethod
    def _annotate_design_output(
        design_output: DesignOutput,
        variant: DesignTeacherVariant,
        context: DesignPolicyContext,
    ) -> DesignOutput:
        variant_id = float(VARIANT_ORDER.index(variant))
        scores = {
            **design_output.design_scores,
            "teacher_variant_id": variant_id,
            "teacher_action_count": float(len(design_output.design_actions)),
            "fixed_simple_p1": 1.0 if context.task_spec.task_type == TaskType.OBJECT_GRASP_CARRY else 0.0,
        }
        return replace(design_output, design_scores=scores)
