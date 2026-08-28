#!/usr/bin/env python3
from __future__ import annotations

"""Run protected C3 with only pi_L joint and normal-compression actions."""

from pathlib import Path
import runpy
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_c3_action_contract import (  # noqa: E402
    ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
)
from amsrr.training.order9_r1_pi_l_action_diagnostic import (  # noqa: E402
    Order9R1JointCompressionProjectionRecorder,
    annotate_order9_r1_joint_compression_rollout,
)
from amsrr.training import order9_tensor_pi_l_runtime  # noqa: E402
from amsrr.utils.hashing import hash_file  # noqa: E402

PROTECTED_ROLLOUT = REPOSITORY / "scripts/order9_vectorized_isaac_rollout.py"
PROTECTED_ROLLOUT_SHA256 = (
    "e813f950dde32818b7375949bfb58baa983dde760e58da22cfbb79cc33b764c2"
)


def _value(arguments: list[str], option: str) -> str:
    matches = [
        index for index, value in enumerate(arguments) if value == option
    ]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise ValueError(f"R1 diagnostic requires exactly one {option}")
    return arguments[matches[0] + 1]


def main() -> int:
    arguments = list(sys.argv[1:])
    if hash_file(PROTECTED_ROLLOUT) != PROTECTED_ROLLOUT_SHA256:
        raise SchemaValidationError("protected C3 rollout bytes changed")
    if _value(arguments, "--c3-action-contract") != (
        ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
    ):
        raise ValueError(
            "R1 diagnostic requires the promoted contact-space contract"
        )
    if "--formal-phase-zero-start" not in arguments:
        raise ValueError("R1 diagnostic requires formal phase-zero execution")
    for forbidden in (
        "--diagnostic-action-ablation",
        "--diagnostic-nominal-qpid-only",
        "--diagnostic-contact-normal-residual-sweep-mm",
        "--diagnostic-virtual-contact-additional-lead-sweep-mm",
    ):
        if forbidden in arguments:
            raise ValueError(f"R1 diagnostic cannot combine {forbidden}")
    output_raw = Path(_value(arguments, "--output-raw")).resolve()
    evaluation_jsonl = Path(_value(arguments, "--evaluation-jsonl")).resolve()
    generation_id = _value(arguments, "--generation-id")
    if not generation_id.startswith("r1_diagnostic:joint_compression_only:"):
        raise ValueError("R1 diagnostic generation id is not isolated")

    original = order9_tensor_pi_l_runtime.apply_order9_contact_space_action
    recorder = Order9R1JointCompressionProjectionRecorder(original)
    order9_tensor_pi_l_runtime.apply_order9_contact_space_action = recorder
    exit_code = 1
    previous_argv = sys.argv
    try:
        sys.argv = [str(PROTECTED_ROLLOUT), *arguments]
        try:
            runpy.run_path(str(PROTECTED_ROLLOUT), run_name="__main__")
        except SystemExit as error:
            exit_code = int(error.code or 0)
    finally:
        sys.argv = previous_argv
        order9_tensor_pi_l_runtime.apply_order9_contact_space_action = original
    if exit_code != 0:
        return exit_code
    annotate_order9_r1_joint_compression_rollout(
        raw_path=output_raw,
        episode_path=evaluation_jsonl,
        recorder=recorder,
        wrapper_path=Path(__file__).resolve(),
        protected_rollout_path=PROTECTED_ROLLOUT,
        protected_rollout_sha256=PROTECTED_ROLLOUT_SHA256,
    )
    print(
        "ORDER9_R1_JOINT_COMPRESSION_DIAGNOSTIC=" + str(output_raw),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
