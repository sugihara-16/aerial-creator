#!/usr/bin/env python3
from __future__ import annotations

"""Create diagnostic line-search checkpoints between two compatible pi_L models."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    order9_state_dict_hash,
    save_order9_policy_checkpoint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, action="append", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    alphas = tuple(float(value) for value in args.alpha)
    if any(not 0.0 < value < 1.0 for value in alphas):
        raise ValueError("interpolation alpha must lie in (0, 1)")
    if len(alphas) != len(set(alphas)):
        raise ValueError("interpolation alphas must be unique")

    parent = load_order9_policy_checkpoint(
        args.parent_checkpoint,
        device="cpu",
        expected_family=Order9PolicyFamily.PI_L,
    )
    candidate = load_order9_policy_checkpoint(
        args.candidate_checkpoint,
        device="cpu",
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=parent.metadata.curriculum_schedule_hash,
    )
    if (
        type(parent.model) is not type(candidate.model)
        or parent.metadata.model_config_hash
        != candidate.metadata.model_config_hash
        or parent.metadata.physical_model_hash
        != candidate.metadata.physical_model_hash
    ):
        raise SchemaValidationError("interpolation endpoint contracts differ")
    parent_state = parent.model.state_dict()
    candidate_state = candidate.model.state_dict()
    if parent_state.keys() != candidate_state.keys():
        raise SchemaValidationError("interpolation state_dict keys differ")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for alpha in alphas:
        model = load_order9_policy_checkpoint(
            args.parent_checkpoint,
            device="cpu",
            expected_family=Order9PolicyFamily.PI_L,
        ).model
        interpolated = {}
        for name, parent_value in parent_state.items():
            candidate_value = candidate_state[name]
            if parent_value.shape != candidate_value.shape:
                raise SchemaValidationError(
                    f"interpolation tensor shape differs: {name}"
                )
            if parent_value.is_floating_point():
                interpolated[name] = torch.lerp(
                    parent_value, candidate_value, alpha
                )
            else:
                if not torch.equal(parent_value, candidate_value):
                    raise SchemaValidationError(
                        f"interpolation discrete tensor differs: {name}"
                    )
                interpolated[name] = parent_value.clone()
        model.load_state_dict(interpolated, strict=True)
        label = f"alpha_{int(round(alpha * 1000)):04d}"
        metadata = replace(
            candidate.metadata,
            state_dict_hash=order9_state_dict_hash(model.state_dict()),
            parent_checkpoint_sha256=parent.sha256,
            metrics={"diagnostic_interpolation_alpha": alpha},
            metadata={
                **candidate.metadata.metadata,
                "acceptance_eligible": False,
                "diagnostic_checkpoint_interpolation": True,
                "interpolation_alpha": alpha,
                "interpolation_parent_sha256": parent.sha256,
                "interpolation_candidate_sha256": candidate.sha256,
            },
        )
        path = output / label / "checkpoint.pt"
        digest = save_order9_policy_checkpoint(
            path, model=model, metadata=metadata
        )
        print(f"{label} {digest} {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
