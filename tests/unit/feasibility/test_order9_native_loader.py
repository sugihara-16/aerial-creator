from __future__ import annotations

import pytest

import amsrr.feasibility.order9_native_loader as native_loader
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionObject,
)


def test_native_loader_rejects_stale_or_unbound_extension(
    tmp_path,
    monkeypatch,
) -> None:
    extension = tmp_path / "_order9_posture_native.so"
    extension.write_bytes(b"not-a-real-extension")
    extension.with_name(
        extension.name + ".build-identity"
    ).write_text("0" * 64 + "\n" + "1" * 64 + "\n", encoding="utf-8")
    monkeypatch.setenv(
        native_loader.ORDER9_POSTURE_NATIVE_PATH_ENV,
        str(extension),
    )
    monkeypatch.setattr(native_loader, "_LOADED_MODULE", None)

    with pytest.raises(
        native_loader.Order9NativePostureIKUnavailable,
        match="stale",
    ):
        native_loader.order9_posture_native_path()


def test_posture_collision_object_rejects_nonpositive_box_size() -> None:
    with pytest.raises(ValueError, match="three positive"):
        Order9PostureCollisionObject(
            object_id="payload",
            size_m=(0.30, 0.0, 0.15),
        )
