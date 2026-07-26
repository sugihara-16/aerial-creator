#!/usr/bin/env python3
from __future__ import annotations

"""Serve the local C3 viewer and persist accept/reject/recalculate actions."""

import argparse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import parse_qs, quote, urlparse
import webbrowser


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.visualization.order9_c3_review import (  # noqa: E402
    order9_c3_review_state,
    record_order9_c3_review,
    reset_order9_c3_review_after_recalculation,
)


DEFAULT_ROOT = "artifacts/p4_full/order9/c3a_curation_pilot_v2"
API_PATH = "/api/order9-c3-review"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the review index in the default browser.",
    )
    parser.add_argument(
        "--recompute-timeout-s",
        type=float,
        default=300.0,
    )
    return parser


class Order9C3ReviewServer(ThreadingHTTPServer):
    curation_root: Path
    recompute_timeout_s: float


def _handler(root: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != API_PATH:
                super().do_GET()
                return
            values = parse_qs(parsed.query)
            case_id = values.get("case_id", [""])[0]
            try:
                state = order9_c3_review_state(root, case_id)
                self._json(HTTPStatus.OK, {"ok": True, **state})
            except (
                FileNotFoundError,
                KeyError,
                SchemaValidationError,
                ValueError,
            ) as error:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": str(error)},
                )

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != API_PATH:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("review request body size is invalid")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("review request must be a JSON object")
                case_id = str(payload.get("case_id", ""))
                action = str(payload.get("action", ""))
                note = str(payload.get("note", ""))
                expected = payload.get("scene_sha256")
                expected_sha256 = (
                    None if expected is None else str(expected)
                )
                if action == "recalculate":
                    response = self._recalculate(
                        case_id=case_id,
                        note=note,
                        expected_scene_sha256=expected_sha256,
                    )
                else:
                    record = record_order9_c3_review(
                        root,
                        case_id=case_id,
                        action=action,
                        note=note,
                        expected_scene_sha256=expected_sha256,
                    )
                    response = {"ok": True, **record}
                self._json(HTTPStatus.OK, response)
            except (
                FileNotFoundError,
                KeyError,
                SchemaValidationError,
                subprocess.SubprocessError,
                ValueError,
            ) as error:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": str(error)},
                )

        def _recalculate(
            self,
            *,
            case_id: str,
            note: str,
            expected_scene_sha256: str | None,
        ) -> dict[str, object]:
            current = order9_c3_review_state(root, case_id)
            if (
                expected_scene_sha256 is not None
                and expected_scene_sha256 != current["scene_sha256"]
            ):
                raise SchemaValidationError(
                    "recalculation request targets a stale scene"
                )
            next_index = int(current["recompute_index"]) + 1
            record_order9_c3_review(
                root,
                case_id=case_id,
                action="recalculate",
                note=note,
                expected_scene_sha256=expected_scene_sha256,
                recompute_index=next_index,
            )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            config_path = Path(str(manifest["source_config_path"]))
            if not config_path.is_absolute():
                config_path = REPOSITORY_ROOT / config_path
            command = [
                sys.executable,
                str(
                    REPOSITORY_ROOT
                    / "scripts/order9_prepare_c3a_curation_pilot.py"
                ),
                "--config",
                str(config_path),
                "--output",
                str(root),
                "--recompute-case",
                case_id,
                "--recompute-index",
                str(next_index),
                "--skip-screenshots",
            ]
            result = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
                timeout=float(self.server.recompute_timeout_s),
                check=False,
            )
            if result.returncode != 0:
                raise subprocess.SubprocessError(
                    "collision-aware recomputation failed: "
                    + (result.stderr or result.stdout)[-2000:]
                )
            state = order9_c3_review_state(root, case_id)
            state = reset_order9_c3_review_after_recalculation(
                root,
                case_id=case_id,
                note=note,
                recompute_index=int(state["recompute_index"]),
            )
            return {
                "ok": True,
                **state,
                "note": note,
                "reload_url": (
                    f"/cases/{quote(case_id)}/final_grasp_mesh.html"
                    f"?review_epoch={time.time_ns()}"
                ),
            }

        def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = (
                json.dumps(payload, sort_keys=True) + "\n"
            ).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> int:
    args = _parser().parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    root = root.resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(root / "manifest.json")
    server = Order9C3ReviewServer(
        (str(args.host), int(args.port)),
        _handler(root),
    )
    server.curation_root = root
    server.recompute_timeout_s = float(args.recompute_timeout_s)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/index.html"
    print(
        "ORDER9_C3_REVIEW_SERVER="
        + json.dumps(
            {"root": str(root), "url": url},
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
