from __future__ import annotations

"""Low-load GUI replay for the 50 s, +/-0.5 Nm, no-safe-hold diagnostic."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from order8_cone_proxy_lift_replay_gui import main as replay_main


SOURCE_REPORT = REPO_ROOT / (
    "artifacts/p4_full/order8_natural_contact/diagnostics/"
    "cone_proxy_pad_tau0p5_all_safehold_v379_50s.json"
)
TRACE_PATH = REPO_ROOT / (
    "artifacts/p4_full/order8_natural_contact/diagnostics/"
    "cone_proxy_pad_tau0p5_all_safehold_v379_50s_state_trace.json"
)


def main(argv: list[str] | None = None) -> int:
    user_argv = list(sys.argv[1:] if argv is None else argv)
    return replay_main(
        [
            "--source-report",
            str(SOURCE_REPORT),
            "--trace-path",
            str(TRACE_PATH),
            *user_argv,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
