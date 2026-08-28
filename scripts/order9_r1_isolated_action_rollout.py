#!/usr/bin/env python3
from __future__ import annotations

"""Run protected C3 with one isolated pi_L correction path."""

from pathlib import Path
import sys
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training import order9_tensor_pi_l_runtime  # noqa: E402
from amsrr.training.order9_c3_action_contract import (  # noqa: E402
    ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
)
from amsrr.training.order9_r1_pi_l_isolated_action_diagnostic import (  # noqa: E402
    ORDER9_R1_ISOLATED_ACTION_MODES,
    Order9R1IsolatedActionMode,
    Order9R1IsolatedProjectionRecorder,
    annotate_order9_r1_isolated_action_rollout,
    protected_rollout_source_without_process_exit,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

PROTECTED_ROLLOUT = REPOSITORY / "scripts/order9_vectorized_isaac_rollout.py"
PROTECTED_ROLLOUT_SHA256 = (
    "e813f950dde32818b7375949bfb58baa983dde760e58da22cfbb79cc33b764c2"
)
MODE_OPTION = "--r1-isolated-action-mode"


def _value(arguments: list[str], option: str) -> str:
    matches = [
        index for index, value in enumerate(arguments) if value == option
    ]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise ValueError(
            f"R1 isolated diagnostic requires exactly one {option}"
        )
    return arguments[matches[0] + 1]


def _remove_value(arguments: list[str], option: str) -> tuple[str, list[str]]:
    value = _value(arguments, option)
    index = arguments.index(option)
    return value, arguments[:index] + arguments[index + 2 :]


def _close_simulation_application(namespace: dict[str, Any]) -> None:
    simulation_app = namespace.get("simulation_app")
    if simulation_app is not None:
        simulation_app.close()


def main() -> int:
    mode_text, arguments = _remove_value(list(sys.argv[1:]), MODE_OPTION)
    if mode_text not in ORDER9_R1_ISOLATED_ACTION_MODES:
        raise ValueError(f"unsupported R1 isolated action mode: {mode_text}")
    mode: Order9R1IsolatedActionMode = mode_text  # type: ignore[assignment]
    if hash_file(PROTECTED_ROLLOUT) != PROTECTED_ROLLOUT_SHA256:
        raise SchemaValidationError("protected C3 rollout bytes changed")
    if _value(arguments, "--c3-action-contract") != (
        ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
    ):
        raise ValueError(
            "R1 isolated diagnostic requires the promoted contact-space contract"
        )
    if "--formal-phase-zero-start" not in arguments:
        raise ValueError(
            "R1 isolated diagnostic requires formal phase-zero execution"
        )
    for forbidden in (
        "--diagnostic-action-ablation",
        "--diagnostic-nominal-qpid-only",
        "--diagnostic-contact-normal-residual-sweep-mm",
        "--diagnostic-virtual-contact-additional-lead-sweep-mm",
    ):
        if forbidden in arguments:
            raise ValueError(
                f"R1 isolated diagnostic cannot combine {forbidden}"
            )
    output_raw = Path(_value(arguments, "--output-raw")).resolve()
    evaluation_jsonl = Path(_value(arguments, "--evaluation-jsonl")).resolve()
    generation_id = _value(arguments, "--generation-id")
    if not generation_id.startswith(f"r1_diagnostic:{mode}:"):
        raise ValueError("R1 isolated diagnostic generation id differs")

    original = order9_tensor_pi_l_runtime.apply_order9_contact_space_action
    recorder = Order9R1IsolatedProjectionRecorder(original, mode=mode)
    order9_tensor_pi_l_runtime.apply_order9_contact_space_action = recorder
    previous_argv = sys.argv
    namespace: dict[str, Any] = {
        "__file__": str(PROTECTED_ROLLOUT),
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
    }
    source = protected_rollout_source_without_process_exit(
        PROTECTED_ROLLOUT.read_text(encoding="utf-8")
    )
    try:
        sys.argv = [str(PROTECTED_ROLLOUT), *arguments]
        exec(compile(source, str(PROTECTED_ROLLOUT), "exec"), namespace)
        exit_code = int(namespace.get("_exit_code", 1))
        if exit_code != 0:
            return exit_code
        annotate_order9_r1_isolated_action_rollout(
            raw_path=output_raw,
            episode_path=evaluation_jsonl,
            recorder=recorder,
            wrapper_path=Path(__file__).resolve(),
            protected_rollout_path=PROTECTED_ROLLOUT,
            protected_rollout_sha256=PROTECTED_ROLLOUT_SHA256,
        )
    finally:
        sys.argv = previous_argv
        order9_tensor_pi_l_runtime.apply_order9_contact_space_action = original
        _close_simulation_application(namespace)
    print(
        "ORDER9_R1_ISOLATED_ACTION_DIAGNOSTIC=" + str(output_raw),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
