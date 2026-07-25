from __future__ import annotations

"""Load the host-built Order 9 posture IK extension without ABI guessing."""

import importlib.machinery
import importlib.util
import hashlib
import os
from pathlib import Path
from types import ModuleType


ORDER9_POSTURE_NATIVE_MODULE_NAME = "_order9_posture_native"
ORDER9_POSTURE_NATIVE_PATH_ENV = "AMSRR_ORDER9_POSTURE_NATIVE_PATH"

_LOADED_MODULE: ModuleType | None = None


class Order9NativePostureIKUnavailable(RuntimeError):
    """The platform-specific production posture IK extension is unavailable."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_native_build_directory() -> Path:
    return repository_root() / "build" / "order9_posture_native"


def native_source_path() -> Path:
    return (
        repository_root()
        / "amsrr"
        / "feasibility"
        / "native"
        / "order9_posture_native.cpp"
    )


def native_build_script_path() -> Path:
    return repository_root() / "scripts" / "build_order9_posture_native.sh"


def _candidate_paths() -> tuple[Path, ...]:
    configured = os.environ.get(ORDER9_POSTURE_NATIVE_PATH_ENV)
    if configured:
        return (Path(configured).expanduser().resolve(),)
    directory = default_native_build_directory()
    return tuple(
        directory / f"{ORDER9_POSTURE_NATIVE_MODULE_NAME}{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def order9_posture_native_path() -> Path:
    for candidate in _candidate_paths():
        if candidate.is_file():
            _validate_build_identity(candidate)
            return candidate
    searched = ", ".join(str(value) for value in _candidate_paths())
    raise Order9NativePostureIKUnavailable(
        "Order 9 native posture IK extension was not found. Build it with "
        f"{native_build_script_path()} (searched: {searched})"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_build_identity(extension_path: Path) -> None:
    identity_path = Path(f"{extension_path}.build-identity")
    if not identity_path.is_file():
        raise Order9NativePostureIKUnavailable(
            "Order 9 native posture IK build identity is missing for "
            f"{extension_path}; rebuild with {native_build_script_path()}"
        )
    values = identity_path.read_text(encoding="utf-8").splitlines()
    if len(values) != 2:
        raise Order9NativePostureIKUnavailable(
            f"invalid Order 9 native build identity: {identity_path}"
        )
    expected = (_sha256(native_source_path()), _sha256(native_build_script_path()))
    if tuple(values) != expected:
        raise Order9NativePostureIKUnavailable(
            "Order 9 native posture IK extension is stale relative to its "
            f"source/build script; rebuild with {native_build_script_path()}"
        )


def load_order9_posture_native() -> ModuleType:
    global _LOADED_MODULE
    if _LOADED_MODULE is not None:
        return _LOADED_MODULE
    path = order9_posture_native_path()
    spec = importlib.util.spec_from_file_location(
        ORDER9_POSTURE_NATIVE_MODULE_NAME,
        path,
    )
    if spec is None or spec.loader is None:
        raise Order9NativePostureIKUnavailable(
            f"could not create an extension loader for {path}"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError) as error:
        raise Order9NativePostureIKUnavailable(
            f"could not load Order 9 native posture IK extension {path}: "
            f"{error}"
        ) from error
    _LOADED_MODULE = module
    return module


def order9_posture_native_available() -> bool:
    try:
        load_order9_posture_native()
    except Order9NativePostureIKUnavailable:
        return False
    return True


__all__ = [
    "ORDER9_POSTURE_NATIVE_MODULE_NAME",
    "ORDER9_POSTURE_NATIVE_PATH_ENV",
    "Order9NativePostureIKUnavailable",
    "default_native_build_directory",
    "load_order9_posture_native",
    "native_build_script_path",
    "native_source_path",
    "order9_posture_native_available",
    "order9_posture_native_path",
    "repository_root",
]
