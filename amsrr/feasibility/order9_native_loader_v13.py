from __future__ import annotations

"""Activate the hash-validated R1 v13 exact-margin native extension."""

import hashlib
import importlib.machinery
import os
from pathlib import Path

from amsrr.feasibility import order9_native_loader as base_loader

ORDER9_POSTURE_NATIVE_V13_VERSION = "order9_posture_native_exact_margin_v13"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def order9_posture_native_v13_path() -> Path:
    root = base_loader.repository_root()
    directory = root / "build/order9_posture_native_v13"
    candidates = tuple(
        directory / f"{base_loader.ORDER9_POSTURE_NATIVE_MODULE_NAME}{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    extension = next((path for path in candidates if path.is_file()), None)
    if extension is None:
        raise base_loader.Order9NativePostureIKUnavailable(
            "R1 v13 native extension is missing; run "
            f"{root / 'scripts/build_order9_posture_native_v13.sh'}"
        )
    identity = Path(f"{extension}.v13-build-identity")
    generated = directory / "order9_posture_native_v13.cpp"
    build_script = root / "scripts/build_order9_posture_native_v13.sh"
    values = (
        identity.read_text(encoding="utf-8").splitlines() if identity.is_file() else []
    )
    expected = [
        ORDER9_POSTURE_NATIVE_V13_VERSION,
        _sha256(base_loader.native_source_path()),
        _sha256(build_script),
        _sha256(generated) if generated.is_file() else "",
    ]
    if values != expected:
        raise base_loader.Order9NativePostureIKUnavailable(
            "R1 v13 native extension build identity differs; rebuild with "
            f"{build_script}"
        )
    # The compatibility identity is also checked by the existing loader.
    base_loader._validate_build_identity(extension)
    return extension


def activate_order9_posture_native_v13() -> Path:
    """Select v13 before the process loads any posture native extension."""

    if base_loader._LOADED_MODULE is not None:
        raise base_loader.Order9NativePostureIKUnavailable(
            "R1 v13 native extension must be activated before native import"
        )
    path = order9_posture_native_v13_path()
    os.environ[base_loader.ORDER9_POSTURE_NATIVE_PATH_ENV] = str(path)
    return path


__all__ = [
    "ORDER9_POSTURE_NATIVE_V13_VERSION",
    "activate_order9_posture_native_v13",
    "order9_posture_native_v13_path",
]
