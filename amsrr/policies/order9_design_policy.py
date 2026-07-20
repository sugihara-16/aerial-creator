from __future__ import annotations

"""Masked autoregressive learned pi_D for Order 9."""

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.policies.design_candidate_generator import DesignActionCandidate
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.order9_design_grammar import (
    Order9DesignGrammar,
    Order9PartialDesignState,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import DesignActionType, DesignOutput
from amsrr.schemas.task_spec import TaskType
from amsrr.utils.hashing import stable_hash


ORDER9_AUTOREGRESSIVE_PI_D_VERSION = "order9_masked_autoregressive_pi_d_v1"
DESIGN_ACTION_TYPES = tuple(DesignActionType)
CONTACT_MODE_VALUES = (
    "grasp",
    "support",
    "push",
    "latch",
    "perch",
    "slide",
    "stick",
    "free_flight",
    "body_contact",
    "tool",
)


@dataclass(frozen=True)
class Order9DesignPolicyConfig:
    context_feature_dim: int = 48
    candidate_feature_dim: int = 40
    d_model: int = 96
    maximum_design_steps: int = 256

    def validate(self) -> None:
        if min(
            self.context_feature_dim,
            self.candidate_feature_dim,
            self.d_model,
            self.maximum_design_steps,
        ) <= 0:
            raise ValueError("Order 9 pi_D dimensions and step limit must be positive")


@dataclass
class Order9PiDStepOutput:
    logits: torch.Tensor
    action_mask: torch.Tensor
    value: torch.Tensor
    candidate_embeddings: torch.Tensor
    next_context_embedding: torch.Tensor


@dataclass
class Order9PiDActionEvaluation:
    selected_index: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class Order9AutoregressiveDesignPolicy(nn.Module):
    """Score grammar candidates and sample one graph edit at a time."""

    def __init__(self, config: Order9DesignPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or Order9DesignPolicyConfig()
        self.config.validate()
        cfg = self.config
        self.context_projection = nn.Sequential(
            nn.Linear(cfg.context_feature_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.SiLU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(cfg.candidate_feature_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.SiLU(),
        )
        self.history_cell = nn.GRUCell(cfg.d_model, cfg.d_model)
        self.state_fusion = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.score_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )

    def initial_history(
        self,
        *,
        batch_size: int = 1,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            (batch_size, self.config.d_model),
            dtype=parameter.dtype,
            device=parameter.device,
        )

    def forward_step(
        self,
        context: DesignPolicyContext,
        state: Order9PartialDesignState,
        candidates: Sequence[DesignActionCandidate],
        *,
        history: torch.Tensor | None = None,
    ) -> Order9PiDStepOutput:
        if not candidates:
            raise ValueError("pi_D step requires candidate actions")
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        if history is None:
            history = self.initial_history()
        if tuple(history.shape) != (1, self.config.d_model):
            raise ValueError("pi_D single-step history must have shape [1, d_model]")
        context_features = torch.tensor(
            [_context_features(context, state, self.config.context_feature_dim)],
            dtype=dtype,
            device=device,
        )
        candidate_features = torch.tensor(
            [
                _candidate_features(candidate, self.config.candidate_feature_dim)
                for candidate in candidates
            ],
            dtype=dtype,
            device=device,
        )
        context_embedding = self.context_projection(context_features)
        fused = self.state_fusion(torch.cat((context_embedding, history), dim=-1))
        candidate_embeddings = self.candidate_projection(candidate_features)
        expanded = fused.expand(candidate_embeddings.shape[0], -1)
        logits = self.score_head(
            torch.cat((expanded, candidate_embeddings), dim=-1)
        ).squeeze(-1)
        action_mask = torch.tensor(
            [candidate.valid for candidate in candidates],
            dtype=torch.bool,
            device=device,
        )
        if not bool(action_mask.any().item()):
            raise SchemaValidationError("pi_D grammar produced no valid action")
        logits = logits.masked_fill(~action_mask, -1.0e9)
        return Order9PiDStepOutput(
            logits=logits,
            action_mask=action_mask,
            value=self.value_head(fused).squeeze(-1),
            candidate_embeddings=candidate_embeddings,
            next_context_embedding=fused,
        )

    def sample_step(
        self,
        output: Order9PiDStepOutput,
        *,
        deterministic: bool = False,
    ) -> Order9PiDActionEvaluation:
        distribution = Categorical(logits=output.logits)
        selected = (
            output.logits.argmax().reshape(1)
            if deterministic
            else distribution.sample().reshape(1)
        )
        return Order9PiDActionEvaluation(
            selected_index=selected,
            log_prob=distribution.log_prob(selected.squeeze(0)).reshape(1),
            entropy=distribution.entropy().reshape(1),
            value=output.value,
        )

    def evaluate_selected_step(
        self,
        output: Order9PiDStepOutput,
        selected_index: int | torch.Tensor,
    ) -> Order9PiDActionEvaluation:
        index = torch.as_tensor(
            selected_index, dtype=torch.long, device=output.logits.device
        ).reshape(1)
        if int(index.item()) < 0 or int(index.item()) >= output.logits.numel():
            raise ValueError("pi_D selected action index is out of range")
        if not bool(output.action_mask[int(index.item())].item()):
            raise SchemaValidationError("pi_D selected action is masked")
        distribution = Categorical(logits=output.logits)
        return Order9PiDActionEvaluation(
            selected_index=index,
            log_prob=distribution.log_prob(index.squeeze(0)).reshape(1),
            entropy=distribution.entropy().reshape(1),
            value=output.value,
        )

    def advance_history(
        self,
        output: Order9PiDStepOutput,
        selected_index: int | torch.Tensor,
    ) -> torch.Tensor:
        index = int(torch.as_tensor(selected_index).item())
        selected = output.candidate_embeddings[index].reshape(1, -1)
        return self.history_cell(selected, output.next_context_embedding)

    def propose(
        self,
        context: DesignPolicyContext,
        *,
        deterministic: bool = True,
        checker: FeasibilityChecker | None = None,
    ) -> DesignOutput:
        """Generate one hard-feasible DesignOutput without invoking fallback."""

        grammar = Order9DesignGrammar(context, checker=checker)
        state = grammar.initial_state()
        history = self.initial_history()
        self.eval()
        with torch.no_grad():
            for _ in range(self.config.maximum_design_steps):
                candidates = grammar.candidates(state)
                output = self.forward_step(
                    context, state, candidates, history=history
                )
                decision = self.sample_step(output, deterministic=deterministic)
                index = int(decision.selected_index.item())
                history = self.advance_history(output, index)
                state = grammar.apply(state, candidates[index])
                if state.stopped:
                    design = grammar.build_design_output(state)
                    result = grammar.checker.check_design(
                        design,
                        task_spec=context.task_spec,
                        irg=context.irg,
                        physical_model=context.physical_model,
                    )
                    if not result.feasible:
                        raise SchemaValidationError(
                            "pi_D final hard feasibility recheck failed"
                        )
                    return design
        raise SchemaValidationError("pi_D did not reach STOP within maximum_design_steps")


def design_action_candidate_feature_vector(
    candidate: DesignActionCandidate,
    *,
    width: int = 40,
) -> list[float]:
    return _candidate_features(candidate, width)


def _context_features(
    context: DesignPolicyContext,
    state: Order9PartialDesignState,
    width: int,
) -> list[float]:
    task_types = list(TaskType)
    modes = CONTACT_MODE_VALUES
    envelope = context.interaction_envelope
    required_modes = {mode.value for mode in envelope.required_contact_modes} if envelope else set()
    coverage = {
        slot_id
        for anchor in state.anchors
        for slot_id in anchor.bound_slot_ids
    }
    required_slots = {
        int(node.feature.get("slot_id", node.node_id))
        for node in context.irg.nodes
        if str(node.node_type.value) == "contact_slot"
        and bool(node.feature.get("required", True))
    }
    physical = context.physical_model
    values = [
        *[1.0 if context.task_spec.task_type == task_type else 0.0 for task_type in task_types],
        float(context.task_spec.robot_constraints.min_modules),
        float(context.task_spec.robot_constraints.max_modules),
        float(len(state.module_ids)),
        float(len(state.connected_port_pairs)),
        float(len(state.module_roles)),
        float(len(state.anchors)),
        float(sum(len(anchor.bound_slot_ids) for anchor in state.anchors)),
        1.0 if state.base_module_id is not None else 0.0,
        1.0 if state.control_group_assigned else 0.0,
        float(len(state.action_history)) / 256.0,
        float(len(coverage & required_slots)) / max(1.0, float(len(required_slots))),
        float(physical.aggregate_mass_kg),
        float(len(physical.rotors)),
        float(len(physical.dock_ports)),
        float(sum(rotor.thrust_max_n for rotor in physical.rotors)),
        float(envelope.required_contact_count_range[0]) if envelope else 0.0,
        float(envelope.required_contact_count_range[1]) if envelope else 0.0,
        *[1.0 if mode in required_modes else 0.0 for mode in modes],
        _stable_scalar(context.task_spec.task_id),
        _stable_scalar(context.irg.irg_id),
    ]
    return _pad_or_trim(values, width)


def _candidate_features(
    candidate: DesignActionCandidate,
    width: int,
) -> list[float]:
    action = candidate.action
    params = action.params
    numeric_names = (
        "module_id",
        "edge_id",
        "src_module_id",
        "dst_module_id",
        "src_port_id",
        "dst_port_id",
        "anchor_id",
        "slot_id",
        "surface_port_id",
        "suggested_slot_id",
    )
    values = [
        *[1.0 if action.action_type == action_type else 0.0 for action_type in DESIGN_ACTION_TYPES],
        1.0 if candidate.valid else 0.0,
        float(candidate.score_prior),
        *[_finite_numeric(params.get(name), scale=64.0) for name in numeric_names],
        _stable_scalar(str(params.get("module_type", ""))),
        _stable_scalar(str(params.get("role_id", ""))),
        _stable_scalar(str(params.get("anchor_type", ""))),
        _stable_scalar(str(params.get("group_id", ""))),
        _stable_scalar(candidate.reason_code),
        _stable_scalar(stable_hash(params)),
        float(len(params)),
    ]
    return _pad_or_trim(values, width)


def _finite_numeric(value: object, *, scale: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number / scale
    return 0.0


def _stable_scalar(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _pad_or_trim(values: list[float], width: int) -> list[float]:
    result = [float(value) for value in values[:width]]
    result.extend([0.0] * (width - len(result)))
    return result
