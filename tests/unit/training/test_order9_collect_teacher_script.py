from __future__ import annotations

import json

import pytest

from amsrr.utils.hashing import hash_file
from scripts.order9_collect_teacher import (
    _acquire_collection_lock,
    _validated_generated_usd_path,
)


def test_teacher_collector_output_root_is_exclusively_locked(tmp_path) -> None:
    first = _acquire_collection_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="another Order9 teacher collector"):
            _acquire_collection_lock(tmp_path)
    finally:
        first.close()

    second = _acquire_collection_lock(tmp_path)
    second.close()


def test_generated_usd_path_is_bound_to_reported_bytes(tmp_path) -> None:
    usd_path = tmp_path / "generated" / "holon.usda"
    usd_path.parent.mkdir(parents=True)
    usd_path.write_bytes(b"#usda 1.0\n")
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "report": {
                    "usd_path": str(usd_path),
                    "generated_usd_sha256": hash_file(usd_path),
                }
            }
        ),
        encoding="utf-8",
    )

    assert _validated_generated_usd_path(report_path) == usd_path.resolve()

    usd_path.write_bytes(b"mutated")
    assert _validated_generated_usd_path(report_path) is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"report": {}},
        {"report": {"usd_path": "missing", "generated_usd_sha256": "bad"}},
    ],
)
def test_generated_usd_path_validation_fails_closed(tmp_path, payload) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    assert _validated_generated_usd_path(report_path) is None
