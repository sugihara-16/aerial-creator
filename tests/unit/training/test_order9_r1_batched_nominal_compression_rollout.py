from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError


REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"


def _module():
    spec = importlib.util.spec_from_file_location("order9_r1_batched_wrapper", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batched_environment_spacing_is_origin_packed() -> None:
    module = _module()
    arguments = ["--env-spacing", "3.0", "--other", "value"]

    module._configure_batched_environment_spacing(arguments)

    assert arguments == [
        "--env-spacing",
        str(module.BATCH_ENV_SPACING_M),
        "--other",
        "value",
    ]


def test_batched_environment_spacing_is_added_when_absent() -> None:
    module = _module()
    arguments = ["--other", "value"]

    module._configure_batched_environment_spacing(arguments)

    assert arguments[-2:] == ["--env-spacing", str(module.BATCH_ENV_SPACING_M)]


def test_batched_scene_rejects_more_than_four_candidates(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            {
                "jobs": [
                    {"name": f"candidate-{index}", "argv": ["--placeholder"]}
                    for index in range(5)
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SchemaValidationError,
        match="exceeds the verified four-candidate limit",
    ):
        module._load_jobs(manifest, offset=0, limit=None)
