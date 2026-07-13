from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from amsrr.schemas.common import canonical_json


def stable_hash(data: Any) -> str:
    return sha256(canonical_json(data).encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory_manifest(directory: str | Path) -> str:
    """Hash relative paths and byte hashes for every file below a directory."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory manifest root does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Directory manifest root contains no files: {root}")
    return stable_hash(
        [
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": hash_file(path),
            }
            for path in files
        ]
    )
