#!/usr/bin/env python3
from __future__ import annotations

"""Resume-safe queue for the complete Order 9 C3 nominal trajectory set."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_configuration_space_planner import (
    ORDER9_CONFIGURATION_SPACE_PLANNER_VERSION,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectorySetManifest,
)
from amsrr.utils.hashing import hash_file


QUEUE_VERSION = "order9_c3_nominal_local_contact_queue_v1"
DEFAULT_BUCKET_MANIFEST = (
    "artifacts/p4_full/order9/stages/"
    "c3_pi_l_ppo_arbitrary_morphology/"
    "rollout_buckets_current_lineage_v5/manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    "artifacts/p4_full/order9/stages/"
    "c3_pi_l_ppo_arbitrary_morphology"
)
DEFAULT_REVIEWED_OVERRIDES = (
    "configs/training/order9_c3_reviewed_contact_overrides_5_to_8.json"
)
OUTPUT_NAME_TEMPLATE = (
    "nominal_trajectory_bucket_support_ground_index_{index:02d}_"
    "local_planner_v8"
)
FINAL_MANIFEST_NAME = (
    "nominal_trajectories_current_lineage_local_planner_v8_manifest.json"
)
LIVE_INDEX_NAME = "nominal_trajectories_current_lineage_v1_live.html"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket-manifest", default=DEFAULT_BUCKET_MANIFEST)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--reviewed-contact-overrides",
        default=DEFAULT_REVIEWED_OVERRIDES,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-bucket-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--indices",
        help="Optional comma-separated source indices; defaults to all.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.per_bucket_timeout_s <= 0.0:
        raise ValueError("--per-bucket-timeout-s must be positive")
    bucket_manifest_path = _resolve(args.bucket_manifest)
    bucket_manifest = json.loads(
        bucket_manifest_path.read_text(encoding="utf-8")
    )
    buckets = bucket_manifest.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("bucket manifest contains no buckets")
    selected_indices = (
        list(range(len(buckets)))
        if args.indices is None
        else _parse_indices(args.indices, len(buckets))
    )
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    queue_root = output_root / ".nominal_local_planner_v8_queue"
    logs_root = queue_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    state_path = queue_root / "state.json"
    live_index_path = output_root / LIVE_INDEX_NAME
    reviewed_path = _resolve(args.reviewed_contact_overrides)
    reviewed_payload = json.loads(reviewed_path.read_text(encoding="utf-8"))
    reviewed_ids = {
        str(entry["bucket_id"])
        for entry in reviewed_payload.get("entries", [])
    }
    rows: dict[int, dict[str, Any]] = {}
    for index, bucket in enumerate(buckets):
        output_dir = output_root / OUTPUT_NAME_TEMPLATE.format(index=index)
        complete = _completed_output(
            output_dir,
            expected_bucket_id=str(bucket["bucket_id"]),
        )
        rows[index] = {
            "index": index,
            "bucket_id": str(bucket["bucket_id"]),
            "split": str(bucket["split"]),
            "module_count": int(bucket["module_count"]),
            "output_dir": str(output_dir.relative_to(REPOSITORY_ROOT)),
            "status": "accepted" if complete else "pending",
            "wall_time_s": None,
            "message": "existing validated output" if complete else "",
        }
    lock = threading.Lock()

    def publish() -> None:
        with lock:
            _write_queue_state(
                state_path,
                rows=rows,
                bucket_manifest_path=bucket_manifest_path,
                selected_indices=selected_indices,
            )
            _write_live_index(live_index_path, rows=rows)

    publish()
    pending = [
        index
        for index in selected_indices
        if rows[index]["status"] != "accepted"
    ]
    print(
        "ORDER9_C3_QUEUE_START="
        f"selected={len(selected_indices)} pending={len(pending)} "
        f"workers={args.workers} state={state_path}",
        flush=True,
    )

    def run_index(index: int) -> tuple[int, str, float, str]:
        bucket = buckets[index]
        bucket_id = str(bucket["bucket_id"])
        output_dir = output_root / OUTPUT_NAME_TEMPLATE.format(index=index)
        log_path = logs_root / f"index_{index:02d}.log"
        with lock:
            rows[index]["status"] = "running"
            rows[index]["message"] = "generation"
            _write_queue_state(
                state_path,
                rows=rows,
                bucket_manifest_path=bucket_manifest_path,
                selected_indices=selected_indices,
            )
            _write_live_index(live_index_path, rows=rows)
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/order9_prepare_c3_nominal_trajectories.py"),
            "--bucket-manifest",
            str(bucket_manifest_path),
            "--bucket-indices",
            str(index),
            "--output",
            str(output_dir),
            "--workers",
            "1",
        ]
        if bucket_id in reviewed_ids:
            command.extend(
                ["--reviewed-contact-overrides", str(reviewed_path)]
            )
        else:
            command.append("--reselect-contacts")
        started = time.perf_counter()
        status, generation_output = _run_bounded(
            command,
            timeout_s=float(args.per_bucket_timeout_s),
        )
        log_parts = [
            f"QUEUE_VERSION={QUEUE_VERSION}",
            f"COMMAND={json.dumps(command)}",
            generation_output,
        ]
        if status != "accepted":
            _atomic_write_text(log_path, "\n".join(log_parts) + "\n")
            return index, status, time.perf_counter() - started, generation_output[-2000:]
        validation_command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/order9_validate_c3_nominal_collision.py"),
            str(output_dir / "manifest.json"),
            "--output",
            str(output_dir / "collision_validation.json"),
        ]
        validation_status, validation_output = _run_bounded(
            validation_command,
            timeout_s=min(float(args.per_bucket_timeout_s), 900.0),
        )
        log_parts.extend(
            [
                f"VALIDATION_COMMAND={json.dumps(validation_command)}",
                validation_output,
            ]
        )
        _atomic_write_text(log_path, "\n".join(log_parts) + "\n")
        if validation_status != "accepted":
            return (
                index,
                f"validation_{validation_status}",
                time.perf_counter() - started,
                validation_output[-2000:],
            )
        if not _completed_output(output_dir, expected_bucket_id=bucket_id):
            return (
                index,
                "invalid_output",
                time.perf_counter() - started,
                "post-generation identity or validation check failed",
            )
        return index, "accepted", time.perf_counter() - started, ""

    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_index, index): index for index in pending
            }
            for completed_count, future in enumerate(
                as_completed(futures), start=1
            ):
                index, status, wall_s, message = future.result()
                with lock:
                    rows[index]["status"] = status
                    rows[index]["wall_time_s"] = wall_s
                    rows[index]["message"] = message
                    _write_queue_state(
                        state_path,
                        rows=rows,
                        bucket_manifest_path=bucket_manifest_path,
                        selected_indices=selected_indices,
                    )
                    _write_live_index(live_index_path, rows=rows)
                print(
                    "ORDER9_C3_QUEUE_PROGRESS="
                    f"{completed_count}/{len(pending)} index={index:02d} "
                    f"bucket={rows[index]['bucket_id']} status={status} "
                    f"wall_s={wall_s:.3f}",
                    flush=True,
                )
    selected_failures = [
        index
        for index in selected_indices
        if rows[index]["status"] != "accepted"
    ]
    all_failures = [
        index for index, row in rows.items() if row["status"] != "accepted"
    ]
    if not all_failures:
        final_manifest = _write_final_manifest(
            output_root,
            rows=rows,
            bucket_manifest_path=bucket_manifest_path,
        )
        publish()
        print(
            "ORDER9_C3_QUEUE_COMPLETE="
            f"manifest={final_manifest} sha256={hash_file(final_manifest)}",
            flush=True,
        )
    elif not selected_failures:
        print(
            "ORDER9_C3_QUEUE_SELECTED_COMPLETE="
            f"remaining_global={all_failures}",
            flush=True,
        )
    else:
        print(
            f"ORDER9_C3_QUEUE_FAILED_INDICES={selected_failures}",
            flush=True,
        )
        return 1
    return 0


def _completed_output(output_dir: Path, *, expected_bucket_id: str) -> bool:
    manifest_path = output_dir / "manifest.json"
    validation_path = output_dir / "collision_validation.json"
    if not manifest_path.is_file() or not validation_path.is_file():
        return False
    try:
        manifest = Order9C3NominalTrajectorySetManifest.from_json(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest.validate()
        if len(manifest.entries) != 1:
            return False
        entry = manifest.entries[0]
        if entry.bucket_id != expected_bucket_id:
            return False
        artifact_path = output_dir / entry.artifact_path
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        windows = artifact.get("windows")
        if not isinstance(windows, list) or not windows:
            return False
        if any(
            window.get("configuration_planner_version")
            != ORDER9_CONFIGURATION_SPACE_PLANNER_VERSION
            for window in windows
        ):
            return False
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        return (
            validation.get("all_accepted") is True
            and validation.get("nominal_manifest_sha256")
            == hash_file(manifest_path)
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _run_bounded(command: list[str], *, timeout_s: float) -> tuple[str, str]:
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        return "timeout", output
    return ("accepted" if process.returncode == 0 else "failed"), output


def _write_queue_state(
    path: Path,
    *,
    rows: dict[int, dict[str, Any]],
    bucket_manifest_path: Path,
    selected_indices: list[int],
) -> None:
    payload = {
        "queue_version": QUEUE_VERSION,
        "planner_version": ORDER9_CONFIGURATION_SPACE_PLANNER_VERSION,
        "bucket_manifest_path": _portable(bucket_manifest_path),
        "bucket_manifest_sha256": hash_file(bucket_manifest_path),
        "selected_indices": selected_indices,
        "updated_unix_s": time.time(),
        "records": [rows[index] for index in sorted(rows)],
    }
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _write_live_index(path: Path, *, rows: dict[int, dict[str, Any]]) -> None:
    accepted = sum(row["status"] == "accepted" for row in rows.values())
    running = sum(row["status"] == "running" for row in rows.values())
    failures = sum(
        row["status"] not in {"accepted", "pending", "running"}
        for row in rows.values()
    )
    body = []
    for index in sorted(rows):
        row = rows[index]
        output_dir = Path(str(row["output_dir"])).name
        viewer = _viewer_relative_path(
            REPOSITORY_ROOT / str(row["output_dir"])
        )
        link = (
            f'<a href="{html.escape(output_dir + "/" + viewer)}">再生</a>'
            if viewer is not None and row["status"] == "accepted"
            else "—"
        )
        wall = row["wall_time_s"]
        body.append(
            "<tr>"
            f"<td>{index:02d}</td>"
            f"<td>{html.escape(str(row['bucket_id']))}</td>"
            f"<td>{int(row['module_count'])}</td>"
            f"<td>{html.escape(str(row['status']))}</td>"
            f"<td>{'—' if wall is None else f'{float(wall):.1f}'}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta http-equiv="refresh" content="20">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>C3 nominal trajectories (live)</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1100px;margin:28px auto;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3db;padding:6px 8px}}
th{{background:#edf2f7;position:sticky;top:0}}code{{background:#eee;padding:2px 4px}}</style></head>
<body><h1>C3 nominal trajectories — 生成状況</h1>
<p>accepted: {accepted}/{len(rows)}、running: {running}、failed: {failures}。20秒ごとに更新します。</p>
<p><code>pi_H teacher -> deterministic IK</code> の理想追従nominal軌道です。
各accepted行は独立convex collision replayにも合格しています。</p>
<table><thead><tr><th>index</th><th>bucket</th><th>modules</th><th>status</th><th>wall [s]</th><th>viewer</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></body></html>"""
    _atomic_write_text(path, document)


def _viewer_relative_path(output_dir: Path) -> str | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["entries"]
        if len(entries) != 1:
            return None
        value = entries[0].get("animation_html_path")
        return str(value) if value else None
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _write_final_manifest(
    output_root: Path,
    *,
    rows: dict[int, dict[str, Any]],
    bucket_manifest_path: Path,
) -> Path:
    entries = []
    component_hashes: dict[str, str] = {}
    validation_hashes: dict[str, str] = {}
    physical_model_hash: str | None = None
    manifest_version: str | None = None
    for index in sorted(rows):
        output_dir = REPOSITORY_ROOT / str(rows[index]["output_dir"])
        component_path = output_dir / "manifest.json"
        component = json.loads(component_path.read_text(encoding="utf-8"))
        component_entry = dict(component["entries"][0])
        physical_model_hash = physical_model_hash or component["physical_model_hash"]
        manifest_version = manifest_version or component["manifest_version"]
        for field in (
            "artifact_path",
            "animation_html_path",
            "animation_scene_path",
        ):
            if component_entry.get(field) is not None:
                component_entry[field] = str(
                    Path(output_dir.name) / str(component_entry[field])
                )
        entries.append(component_entry)
        component_hashes[str(index)] = hash_file(component_path)
        validation_hashes[str(index)] = hash_file(
            output_dir / "collision_validation.json"
        )
    payload = {
        "manifest_version": manifest_version,
        "bucket_manifest_path": _portable(bucket_manifest_path),
        "bucket_manifest_sha256": hash_file(bucket_manifest_path),
        "physical_model_hash": physical_model_hash,
        "entries": entries,
        "index_html_path": LIVE_INDEX_NAME,
        "metadata": {
            "bucket_count": len(entries),
            "source_bucket_indices": sorted(rows),
            "planner_version": ORDER9_CONFIGURATION_SPACE_PLANNER_VERSION,
            "queue_version": QUEUE_VERSION,
            "ideal_tracking_only": True,
            "independent_convex_collision_replay": "accepted_all",
            "component_manifest_sha256_by_index": component_hashes,
            "collision_validation_sha256_by_index": validation_hashes,
        },
    }
    manifest = Order9C3NominalTrajectorySetManifest.from_dict(payload)
    manifest.validate()
    destination = output_root / FINAL_MANIFEST_NAME
    _atomic_write_text(destination, manifest.to_json(indent=2) + "\n")
    return destination


def _parse_indices(raw: str, bucket_count: int) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",")]
    if not values or len(values) != len(set(values)):
        raise ValueError("--indices must be a non-empty unique list")
    if any(value < 0 or value >= bucket_count for value in values):
        raise ValueError("--indices contains an out-of-range value")
    return values


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
